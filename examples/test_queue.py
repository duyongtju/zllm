import multiprocessing
import time

def worker(input_queue, worker_id):
    while True:
        # 从队列中获取对象
        item = input_queue.get()
        if item is None:  # 如果收到None则退出
            print(f"Worker {worker_id} exiting.")
            break
        print(f"Worker {worker_id} got item: {item}")
        time.sleep(1)  # 模拟处理时间

if __name__ == "__main__":
    # 创建输入队列
    input_queue = multiprocessing.Queue()

    # 创建和启动两个子进程
    processes = []
    for i in range(2):
        p = multiprocessing.Process(target=worker, args=(input_queue, i))
        p.start()
        processes.append(p)

    # 主进程向队列中放入对象
    item = "Hello, multiprocessing!"
    input_queue.put(item)
    input_queue.put(item)

    # 给两个子进程发送结束信号
    input_queue.put(None)
    input_queue.put(None)

    # 等待子进程结束
    for p in processes:
        p.join()