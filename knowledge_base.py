"""
知识库
"""
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
def check_md5(md5_str:str):
    """检查传入的md5字符串是否已经被处理过了"""
    if not os.path.exists(config.md5_path):
        #if进入表示文件不存在，那肯定没有处理过这个md5值
        open(config.md5_path, 'w',encoding='utf-8').close()
        return False   #表示MD5未处理过，return Ture表示已经处理过，已有记录
    else:
        for line in open(config.md5_path, 'r',encoding='utf-8').readlines():
            line=line.strip()  #处理字符串前后的空格和回车
            if line == md5_str:
                return True     #已经处理过
        return False

def save_md5(md5_str:str):
    #"""将传入的MD5字符串，记录到文件内保存"""
   with open(config.md5_path, 'a',encoding='utf-8') as f:
       f.write(md5_str+'\n')

def get_string_md5(input_str:str,encoding='utf-8'):
    """将传入的字符串转换为md5字符串"""
    #将字符串转换为bytes字节数组
    str_bytes=input_str.encode(encoding=encoding)
    #创建md5对象
    md5_obj = hashlib.md5()   #得到md5对象
    md5_obj.update(str_bytes) #更新内容(传入即将要转换的字节数组)
    md5_hex=md5_obj.hexdigest()  #得到md5的十六进制字符串
    return md5_hex

class KnowledgeBaseService():
    def __init__(self):
        #如果文件夹不存在，则创建，存在，则跳过
        os.makedirs(config.persist_directory,exist_ok=True)

        self.chroma=Chroma(
            collection_name=config.collection_name,  #统一存放到config_data里
            embedding_function=DashScopeEmbeddings(model='text-embedding-v4'),
            persist_directory=config.persist_directory  #数据本地存储文件夹
        ) #向量存储的实例，Chroma向量库对象

        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len
        )  #文本分割器的对象
    def upload_by_str(self,data:str,filename):
        """将传入的字符串，进行向量化存入向量数据库中"""
        #先得到传入字符串的MD5值
        md5_hex=get_string_md5(data)

        if check_md5(md5_hex):
            return '[跳过],知识已经存在知识库中'
        if len(data) > config.max_spliter_char_number:
            knowledge_chunks: list[str] =self.splitter.split_text(data)
        else:
            knowledge_chunks=[data]  #如果内容很少直接放在列表里，为了统一格式list[str]

        metadata={
            'source':filename,
            'create_time':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  #添加创建数据时间 年月日，时分秒
            'operator':'知更鸟'

        }

        self.chroma.add_texts(      #内容加载到向量数据库中
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks]
        )

        #
        save_md5(md5_hex)

        return '[成功],内容已经成功载入向量库'





if __name__ == '__main__':
   service=KnowledgeBaseService()
