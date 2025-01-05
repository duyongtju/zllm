import asyncio
from dataclasses import dataclass
from multiprocessing.connection import wait
from multiprocessing.process import BaseProcess
import os
import sys
import threading
import multiprocessing
from multiprocessing import Queue
import traceback
from typing import Any, Callable, Dict, Generic, List, Optional, TextIO, TypeVar, Union
import uuid

from zllm.logger import init_logger


logger = init_logger(__name__)

T = TypeVar('T')

_TERMINATE = "TERMINATE"  # sentinel

# ANSI color codes
CYAN = '\033[1;36m'
RESET = '\033[0;0m'

JOIN_TIMEOUT_S = 2

# mp_method = "fork"
mp_method = "spawn"
mp = multiprocessing.get_context(mp_method)


@dataclass
class Result(Generic[T]):

    task_id: uuid.UUID
    value: Optional[T] = None
    exception: Optional[BaseException] = None

class ResultFuture(threading.Event, Generic[T]):

    def __init__(self) -> None:
        super().__init__()
        self.result: Optional[Result[T]] = None

    def set_result(self, result: Result[T]):
        self.result = result
        self.set()

    def get(self) -> T:
        self.wait()
        assert self.result is not None
        if self.result.exception is not None:
            raise self.result.exception
        return self.result.value

def _set_future_result(future: Union[ResultFuture, asyncio.Future],
                       result: Result):
    if isinstance(future, ResultFuture):
        future.set_result(result)
        return
    
    loop = future.get_loop()
    if not loop.is_closed():
        if result.exception is not None:
            loop.call_soon_threadsafe(future.set_exception, result.exception)
        else:
            loop.call_soon_threadsafe(future.set_result, result.value)


class ResultHandler(threading.Thread):

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.result_queue = mp.Queue()
        self.tasks: Dict[uuid.UUID, Union[ResultFuture, asyncio.Future]] = {}

    def run(self):
        for result in iter(self.result_queue.get, _TERMINATE):
            future = self.tasks.pop(result.task_id)
            _set_future_result(future, result)
        for task_id, future in self.tasks.items():
            _set_future_result(
                future, 
                Result(task_id=task_id,
                       exception=ChildProcessError("worker died"))
            )
        
    def close(self):
        self.result_queue.put(_TERMINATE)


class WorkerMonitor(threading.Thread):

    def __init__(self, workers: List['ProcessWorkerWrapper'],
                 result_handler: ResultHandler):
        super().__init__(daemon=True)
        self.workers: List['ProcessWorkerWrapper'] = workers
        self.result_hander = result_handler
        self._close: bool = False

    def run(self):
        dead_sentinels = wait([ worker.process.sentinel for worker in self.workers])
        if not self._close:
            self._close = True

            for worker in self.workers:
                process = worker.process
                if process.sentinel in dead_sentinels:
                    process.join(JOIN_TIMEOUT_S)
                if process.exitcode is not None and process.exitcode != 0:
                    logger.error("Worker %s pid %s died, exit code %s",
                                 process.name, process.pid, process.exitcode)
                
                logger.info("Killing local vLLM worker processes")
                for worker in self.workers:
                    worker.kill_worker()

                self.result_hander.close()

            for worker in self.workers:
                worker.process.join(JOIN_TIMEOUT_S)

    def close(self):
        if self.close:
            return
        self._close = True
        logger.info("Terminating local zLLM worker process")
        for worker in self.workers:
            worker.terminate_worker()
            
        self.result_hander.close()

class ProcessWorkerWrapper:

    def __init__(self, result_handler: ResultHandler,
                 worker_factory: Callable[[], Any]
                 ) -> None:
        self._task_queue = mp.Queue()
        self.result_queue = result_handler.result_queue
        self.tasks = result_handler.tasks
        self.process: BaseProcess = mp.Process(
            target=_run_worker_process,
            name="ZllmWorkerProcess",
            kwargs=dict(
                worker_factory=worker_factory,
                task_queue=self._task_queue,
                result_queue=self.result_queue,
            ),
            daemon=True
        )

        self.process.start()
    
    def _enqueue_task(self, future: Union[ResultFuture, asyncio.Future],
                      method: str,args, kwargs):
        task_id = uuid.uuid4()
        self.tasks[task_id] = future
        try:
            self._task_queue.put((task_id, method, args, kwargs))
        except BaseException as e:
            del self.tasks[task_id]
            raise ChildProcessError("worker died") from e
        
    def execute_method(self, method: str, *args, **kwargs):
        future: ResultFuture = ResultFuture()
        self._enqueue_task(future, method, args, kwargs)
        return future
    
    async def execute_method_async(self, method: str, *args, **kwargs):
        future = asyncio.get_running_loop().create_future()
        self._enqueue_task(future, method, args, kwargs)
        return await future

    def terminate_worker(self):
        try:
            self._task_queue.put(_TERMINATE)
        except ValueError:
            self.process.kill()
        self._task_queue.close()

    def kill_worker(self):
        self._task_queue.close()
        self.process.kill()

def _run_worker_process(
    worker_factory: Callable[[], Any],
    task_queue: Queue,
    result_queue: Queue
) -> None:
    
    process_name = mp.current_process().name
    pid = os.getpid()
    _add_prefix(sys.stdout, process_name, pid)
    _add_prefix(sys.stderr, process_name, pid)

    worker = worker_factory()
    del worker_factory

    logger.info("Worker ready; awaiting task")
    try:
        for items in iter(task_queue.get, _TERMINATE):
            output = None
            exception = None
            task_id, method, args, kwargs = items
            try:
                executor = getattr(worker, method)
                output = executor(*args, **kwargs)
            except BaseException as e:
                tb = traceback.format_exc()
                logger.error(
                    "Exception in worker %s while processing method %s: %s, %s",
                    process_name, method, e, tb)
                exception = e
            result_queue.put(
                Result(task_id=task_id, value=output, exception=exception))
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Worker failed")

    logger.info("Worker exiting")


def _add_prefix(file: TextIO, worker_name: str, pid: int) -> None:
    """Prepend each output line with process-specific prefix"""

    prefix = f"{CYAN}({worker_name} pid={pid}){RESET} "
    file_write = file.write

    def write_with_prefix(s: str):
        if not s:
            return
        if file.start_new_line:  # type: ignore[attr-defined]
            file_write(prefix)
        idx = 0
        while (next_idx := s.find('\n', idx)) != -1:
            next_idx += 1
            file_write(s[idx:next_idx])
            if next_idx == len(s):
                file.start_new_line = True  # type: ignore[attr-defined]
                return
            file_write(prefix)
            idx = next_idx
        file_write(s[idx:])
        file.start_new_line = False  # type: ignore[attr-defined]

    file.start_new_line = True  # type: ignore[attr-defined]
    file.write = write_with_prefix  # type: ignore[method-assign]