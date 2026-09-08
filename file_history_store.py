import os
from langchain_community.chat_message_histories import FileChatMessageHistory


def get_history(session_id:str)->FileChatMessageHistory:
    os.makedirs('./chat_history',exist_ok=True)
    return FileChatMessageHistory(
        file_path=os.path.join('./chat_history',session_id),# 返回真实路径
    )


