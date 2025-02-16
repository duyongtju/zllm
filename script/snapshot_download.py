from huggingface_hub import snapshot_download

snapshot_download(repo_id='yard1/llama-2-7b-sql-lora-test',  # 替换成你要下载的模型或者数据集
                  repo_type='model',                     # model or dataset
                  local_dir='/home/duyong/model-zoos/yard1/llama-2-7b-sql-lora-test',       # 本地目录
                )
                #   use_auth_token='hf_xxx', # 替换 token
                #   resume_download=True)
                