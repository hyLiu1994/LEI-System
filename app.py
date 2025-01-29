from flask import Flask, request, jsonify, send_file
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks import StdOutCallbackHandler

from langchain.llms import OpenAI
import numpy as np
import os

# 添加这些行来设置 Matplotlib 后端
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

from langchain_core.tools.base import BaseTool
from pydantic import Field, BaseModel

from typing import List, Dict
import sys
import tool.operation_data_tool as odt
import tool.statistic_data_tool as sdt
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from utils.DataProcessModule import load_trajectory_dataset_df
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
import uuid


os.environ[
    "OPENAI_API_KEY"] = ""
llm = OpenAI(temperature=0, model="gpt-4o")
llmChat = ChatOpenAI(model="gpt-4o")

dataprocess_agent = None

ALLOWED_COLUMNS = [
    '# Timestamp', 'Type of mobile', 'MMSI', 'Latitude', 'Longitude',
    'Navigational status', 'ROT', 'SOG', 'COG', 'Heading', 'IMO', 'Callsign',
    'Name', 'Ship type', 'Cargo type', 'Width', 'Length',
    'Type of position fixing device', 'Draught', 'Destination', 'ETA',
    'Data source type'
]

def draw_pie_chart(column_name, top_n):
    if column_name not in ALLOWED_COLUMNS:
        raise ValueError(f"Invalid column name: {column_name}. Please choose from {ALLOWED_COLUMNS}")

    # Group by the specified column and count the number of occurrences in each category
    column_counts = df[column_name].value_counts().nlargest(top_n)

    # Plot the pie chart
    plt.figure(figsize=(10, 7))
    column_counts.plot.pie(autopct='%1.1f%%', startangle=140)
    plt.title(f'Top {top_n} Categories by {column_name}')

    # Save the plot as an image
    image_path = os.path.join(app.root_path, 'static', 'pie_chart.png')
    plt.savefig(image_path)
    plt.close()

    return image_path


class DrawPieChartTool(BaseTool):
    top_n: int = Field(5, description="top_n items to draw the image.")  # Default value set to 5
    column_name: str = Field("Ship type",
                             description="The name of the column to update. Must be one of the allowed columns.")  # Default value set to "Ship type"

    def __init__(self):
        super().__init__(name="draw_pie_chart",
                         description="Draw a pie chart of the top N categories by a specified column")

    def _run(self, column_name: str, top_n: int):
        return draw_pie_chart(column_name, top_n)

    def _arun(self, column_name, top_n):
        raise NotImplementedError("Async method not implemented")


app = Flask(__name__)

# 初始化一个空的 DataFrame
df = pd.DataFrame()


@app.route('/api/upload', methods=['POST'])
def upload_file():
    global df, dataprocess_agent
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.csv'):
        # Save the file
        filename = 'uploaded_file.csv'
        file.save(filename)
        # Read the CSV file
        df = pd.read_csv(filename)
        draw_pie_chart_tool = DrawPieChartTool()
        dataprocess_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True,
            return_intermediate_steps=True
        )
        return jsonify({'message': 'File uploaded successfully'}), 200
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/api/data', methods=['GET'])
def get_data(cdf=None):
    if (cdf is None):
        global df
        cdf = df
    # 获取列名
    columns = cdf.columns.tolist() if not cdf.empty else []
    # 将 DataFrame 转换为字典列表，同时处理 NaN 值
    data = cdf.replace({np.nan: None}).to_dict('records') if not cdf.empty else []
    # 返回包含列名和数据的字典
    return jsonify({
        'columns': columns,
        'data': data[:1000]  # 只返回前1000条数据
    })

conversations = {}
@app.route('/api/chat', methods=['POST'])
def chat():
    global dataprocess_agent, df
    data = request.get_json()
    message = data['message']
    response = message
    if dataprocess_agent is not None:
        if "Help" in message:  
            user_id = data.get('session_id')
            message = data.get('message')
            if not user_id or not message:
                return jsonify({"error": "Missing session_id or message"}), 400

            conversation = conversations.get(user_id, [])
            conversation.append({"role": "user", "content": message})
            # 调用模型获取回复
            response_data = get_model_response(conversation)

            conversation.append({"role": "assistant", "content": response_data["response"]})
            conversations[user_id] = conversation

            return jsonify(response_data)
        else:
            response = dataprocess_agent.invoke(message)
            print(response)
            intermediate_steps = response['intermediate_steps']
            response = response['output']

            for step in intermediate_steps:
                print(step[0].tool_input)
                if isinstance(step[0].tool_input, str):
                    try:
                        # 评估操作输入，看是否是DataFrame操作
                        result = eval(step[0].tool_input)
                        if isinstance(result, pd.DataFrame):
                            # 更新全局DataFrame
                            df = result
                            print("DataFrame updated:", df)
                    except Exception as e:
                        print(f"Error evaluating action input: {e}")

            # 在响应中包含更新后的数据
            data_response = get_data()
            response_data = {
                'response': response,
                'data': data_response.json
            }
    else:
        response_data = {'response': "Agent not initialized. Please upload a file first."}

    return jsonify(response_data)

def get_model_response(conversation: List[Dict[str, str]]) -> Dict[str, str]:
    # 绑定工具到模型
    tools = [odt.DeleteGPSPointByMMIDTool, odt.UpdateGPSPointColByIndexTool, sdt.StatisticCategoryByScTool,
             sdt.GpsPositionsByTrajectoryIdsTool]
    llm_with_tools = llmChat.bind_tools(tools)

    messages = [{"role": msg["role"], "content": msg["content"]} for msg in conversation]
    response = llm_with_tools.invoke(messages[-1]["content"])
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call['name'] == "DeleteGPSPointByMMIDTool":
                mmsi = tool_call['args'].get("mmsi")
                if deleteGpsPointByMmsi(mmsi):
                    return {"response": "Delete successful."}
                else:
                    return {"response": f"Record with MMID {mmsi} not found."}

            elif tool_call['name'] == "UpdateGPSPointColByIndexTool":
                try:
                    index = tool_call['args'].get("index")
                    column_name = tool_call['args'].get("column_name")
                    value = tool_call['args'].get("value")

                    # 验证列名
                    odt.UpdateGPSPointColByIndexTool(index=index, column_name=column_name, value=value)

                    if updateGpsPointColByIndex(index, column_name, value):
                        return {"response": "Update successful."}
                    else:
                        return {"response": f"Failed to update record at index {index}."}
                except ValidationError as e:
                    # 返回错误消息
                    return {"response": f"Invalid column name. Allowed columns are: {', '.join(odt.ALLOWED_COLUMNS)}"}
            elif tool_call['name'] == "StatisticCategoryByScTool":
                # 解析工具调用参数
                try:
                    n = tool_call['args'].get("top_n")
                    column_name = tool_call['args'].get("column_name")

                    # 验证列名
                    sdt.StatisticCategoryByScTool(top_n=n, column_name=column_name)

                    key_list, value_list = statisticCategoryBySc(n, column_name)
                    # 生成饼状图
                    image_path = create_pie_chart(value_list, key_list)
                    # 返回包含图像和文本信息的响应
                    if image_path != '':
                        return {
                            'type': 'image',
                            'response': 'Here is the image you requested:',
                            'text': 'Here is the image you requested:',
                            'image_url': image_path
                        }
                    else:
                        return {
                            'type': 'text',
                            'response': 'Sorry, the image is not available.'
                        }
                except ValidationError as e:
                    return {"response": f"Invalid column name. Allowed columns are: {', '.join(sdt.ALLOWED_COLUMNS)}"}
            elif tool_call['name'] == "GpsPositionsByTrajectoryIdsTool":
                # 解析工具调用参数
                trajectory_ids = tool_call['args'].get("trajectory_sequence_nums")
                # 获取轨迹数据
                trajectory_map = gpsPositionsByTrajectoryIdList(trajectory_ids)
                if len(trajectory_map) <= 0:
                    return {'type': 'text', "response": "Trajectory_id is not exist."}
                else:
                    _, one_trajectory_position = next(iter(trajectory_map.items()))  # [(),()]
                    sum_x = sum(pos[0] for pos in one_trajectory_position)
                    sum_y = sum(pos[1] for pos in one_trajectory_position)
                    avg_x = sum_x / len(one_trajectory_position)
                    avg_y = sum_y / len(one_trajectory_position)
                    center = [round(avg_x, 4), round(avg_y, 4)]
                    one_trajectory = [list(map(lambda x: round(x, 4), pos)) for pos in one_trajectory_position]
                    one_trajectory.insert(0, center)
                    return {
                        'type': 'trajectory',
                        'response': 'Here is a trajectory',
                        'text': 'Here is a trajectory',
                        'path': one_trajectory,
                        'center': center,
                        'zoom': 12  # 增加缩放级别以更好地显示伦敦市区
                    }

    return {"response": response.content}


def deleteGpsPointByMmsi(mmsi):
    global df
    df = df[df['MMSI'] != mmsi]
    return True


def updateGpsPointColByIndex(index, column_name, value):
    global df
    # 检查索引是否在 DataFrame 的范围内
    if index in df.index:
        # 使用 .loc 方法更新指定索引和列的值
        df.loc[index, column_name] = value
        return True
    # else:
    # print(f"Warning: Index {index} is out of bounds. No update was made.")
    return False


def create_pie_chart(values, labels):
    # Group by the specified column and count the number of occurrences in each category
    # column_counts = df[column_name].value_counts().nlargest(top_n)

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    unique_id = uuid.uuid4()
    # 将UUID转换成字符串，并去掉其中的破折号
    filename = str(unique_id).replace('-', '')
    # Save the plot as an image
    image_path = os.path.join(app.root_path, 'DataFigure', f'pie_chart_{filename}.png')
    plt.savefig(image_path)
    plt.close()

    return f'/DataFigure/pie_chart_{filename}.png'


def statisticCategoryBySc(n, column_name):
    global df
    mmsi_count_map = df.groupby(column_name).size().to_dict()
    sorted_items = sorted(mmsi_count_map.items(), key=lambda x: x[1], reverse=True)
    # 选取前 n 条数据
    top_n_items = sorted_items[:n]

    # 将结果转换为两个列表
    keyList, valueList = zip(*top_n_items)

    return keyList, valueList


def gpsPositionsByTrajectoryIdList(trajectory_ids):
    ship_data = load_trajectory_dataset_df(df, file_name='aisdk', day='20060302',
                                                                   org_file_dir='./Data')
    res_dict = {key: ship_data[key]["positions"] for key in trajectory_ids if key in ship_data}
    return res_dict

if __name__ == '__main__':
    app.run(threaded=False)  # 设置 threaded=False
