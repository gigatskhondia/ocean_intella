import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from logic import get_static_pipelay_res,MODEL

memory = MemorySaver()

agent_ = create_react_agent(MODEL.openai_client, tools=[get_static_pipelay_res], checkpointer=memory)


if "messages" not in st.session_state:
    st.session_state.messages = [{'role': "system", "content": """This is an offshore pipeline installation engineering 
    agent. It is tasked with calculating static pipelay analysis. When doing static analysis and providing the results,
    it uses paradigm: think, act, observe."""}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello, how can I help you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

with st.chat_message("assistant"):

    stream = agent_.invoke(
        {"input": prompt, "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]},
        {
            # "callbacks":[get_streamlit_cb(st.empty())],
            "configurable": {"thread_id": "abc321"},
        },
    )

    response = list(stream["messages"][len(stream["messages"])-1])[0][1]
    st.write(response)

st.session_state.messages.append({"role": "assistant", "content": response})
