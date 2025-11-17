import json
import requests
from openai import OpenAI
import os
openai_api_key = os.getenv("OPENAI_API_KEY")
weather_api_key = os.getenv("XINZHI_WEATHER_API_KEY")
# print(openai_api_key)
# print(weather_api_key)
#calling_function应用--第一个完整项目
def get_weather(loc):
    url = "https://api.seniverse.com/v3/weather/now.json"
    params = {
        "key": weather_api_key, #填写你的私钥
        "location": loc,
        "language": "zh-Hans",
        "unit": "c",
    }
    response = requests.get(url, params=params)
    temperature = response.json()
    return temperature['results'][0]['now']
def run_conv(messages,
             api_key,
             tools=None,
             functions_list=None,
             model="deepseek-chat"):
    user_messages = messages

    client = OpenAI(api_key=api_key,
                    base_url="https://api.deepseek.com")

    # 如果没有外部函数库，则执行普通的对话任务
    if tools == None:
        response = client.chat.completions.create(
            model=model,
            messages=user_messages
        )
        final_response = response.choices[0].message.content

    # 若存在外部函数库，则需要灵活选取外部函数并进行回答
    else:
        # 创建外部函数库字典
        available_functions = {func.__name__: func for func in functions_list}

        # 创建包含用户问题的message
        messages = user_messages

        # first response
        response = client.chat.completions.create(
            model=model,
            messages=user_messages,
            tools=tools,
        )
        response_message = response.choices[0].message

        # 获取函数名
        function_name = response_message.tool_calls[0].function.name
        # 获取函数对象
        fuction_to_call = available_functions[function_name]
        # 获取函数参数
        function_args = json.loads(response_message.tool_calls[0].function.arguments)

        # 将函数参数输入到函数中，获取函数计算结果
        function_response = fuction_to_call(**function_args)

        # messages中拼接first response消息
        user_messages.append(response_message.model_dump())

        # messages中拼接外部函数输出结果
        user_messages.append(
            {
                "role": "tool",
                "content": json.dumps(function_response),
                "tool_call_id": response_message.tool_calls[0].id
            }
        )

        # 第二次调用模型
        second_response = client.chat.completions.create(
            model=model,
            messages=user_messages)

        # 获取最终结果
        final_response = second_response.choices[0].message.content

    return final_response

messages = [{"role": "user", "content": "请问上海今天天气如何？"}]
get_weather_function = {
    'name': 'get_weather',
    'description': '查询即时天气函数，根据输入的城市名称，查询对应城市的实时天气',
    'parameters': {
        'type': 'object',
        'properties': {  # 参数说明
            'loc': {
                'description': '城市名称',
                'type': 'string'
            }
        },
        'required': ['loc']  # 必备参数
    }
}
tools = [
    {
        "type": "function",
        "function": get_weather_function
    }
]
final_response = run_conv(messages=messages,
         api_key=openai_api_key,
         tools=tools,
         functions_list=[get_weather])
print(final_response)
