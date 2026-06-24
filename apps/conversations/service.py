from config.agent_config import model


async def generate_conversation_title(
    user_query: str,
    answer: str,
) -> str:
    prompt = f"""
请根据下面的首轮问答生成一个简短会话标题。

要求：
- 只输出标题
- 不要超过 20 个中文字符
- 不要使用引号
- 不要使用句号
- 标题要能概括用户主要问题

用户问题：
{user_query}

助手回答：
{answer}
"""

    response = await model.ainvoke(prompt)
    title = response.content.strip()
    return title[:40] or user_query[:20]