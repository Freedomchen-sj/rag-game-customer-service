md5_path='./md5.txt'

#chroma  数据库
collection_name='rag'  #表名
persist_directory='./chroma_db'  #数据库存放路径

#spliter   文本分割器
chunk_size=1000
chunk_overlap=100
separators=['\n\n','\n','',' ',',','.','?','!','，','。','？','！',':','：',';','；']
max_spliter_char_number=1000   #文本分割的阈值

#
top_k=3     #检索返回匹配的文档数量

embedding_model_name='text-embedding-v4'
chat_model_name='qwen3-max'
base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"

session_config={
        'configurable':{
            'session_id':'user_001'
        }
    }