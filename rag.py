import config_data as config
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.runnables import RunnableWithMessageHistory,RunnableLambda
from file_history_store import get_history


def print_prompt(prompt):
    print('###' * 66)
    print(prompt.to_string())
    print('###' * 66)
    return prompt
class RagService(object):
    def __init__(self):

        self.vector_service=VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name),
        )

        self.prompt_template=ChatPromptTemplate.from_messages(
            [
                ('system','以我提供的参考资料为主，回答用户问题.参考资料:\n{context}'),
                ('system','并且我提供用户的历史会话记录，你可以基于历史会话记录回答问题，历史会话记录如下:'),
                MessagesPlaceholder('history'),
                ('human','回答问题:{input}')
            ]
        )

        self.chat_model=ChatOpenAI(
            model=config.chat_model_name,
            base_url=config.base_url
        )

        self.chain=self._get_chain()

    def _get_chain(self):
        """获取最终的执行链"""
        retriever=self.vector_service.get_retriever()

        def format_document(docs:list[Document]):
            if docs is None:
                print('未找到相关资料')
            formatted_str=''
            for doc in docs:
                formatted_str+=f'文档片段:\n{doc.page_content}\n文档元数据:\n{doc.metadata}\n\n'
            return formatted_str

        def format_for_retriever(value:dict)->str:
            return value['input']
        def format_for_prompt_template(value):
            new_value={}
            new_value['input']=value['input']['input']
            new_value['context']=value['context']
            new_value['history']=value['input']['history']
            return new_value

        chain=({'input':RunnablePassthrough(),'context':RunnableLambda(format_for_retriever) | retriever | format_document} |RunnableLambda(format_for_prompt_template) |self.prompt_template | print_prompt | self.chat_model)

        #增强链
        conversation_chain=RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key='input',
            history_messages_key='history'
        )

        return conversation_chain

if __name__ == '__main__':
    #配置session_id
    session_config={
        'configurable':{
            'session_id':'user_001'
        }
    }



