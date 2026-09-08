import time
from rag import RagService
import streamlit as st
import config_data as config

#标题
st.title('GameRobin')
st.divider()    #分隔符

if 'message' not in st.session_state:
    st.session_state['message'] = [{'role':'assistant','content':'你好,我是智能体知更鸟，旨在为游戏玩家提供游戏理解和英雄出装以及玩法思路，帮助您在休闲娱乐时有一个好的游戏体验\n\n(目前支持的游戏有:\n\nMOBA类：王者荣耀|英雄联盟\n\nFPS类:瓦洛兰特|csgo\n\n后续会支持更多游戏，敬请期待)\n\n现在，你有什么游戏想要问我的，尽管提出来吧！'}]

if 'rag' not in st.session_state:
    st.session_state['rag'] = RagService()

for message in st.session_state['message']:
    st.chat_message(message['role']).write(message['content'])

#在页面最下方提供用户输入栏
prompt=st.chat_input()

if prompt:

    #在页面里输出用户的提问
    st.chat_message('user').write(prompt)
    st.session_state['message'].append({'role':'user','content':prompt})

    with st.spinner('AI思考中'):
        res_stream=st.session_state['rag'].chain.stream({'input':prompt},config.session_config)
        res_history=st.chat_message('assistant').write_stream(res_stream)
        st.session_state['message'].append({'role': 'assistant', 'content': res_history})
