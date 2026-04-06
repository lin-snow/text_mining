# 文本信息挖掘概论 - 作业1 数据收集与预处理

## 一、背景简介
文本数据收集是展开文本信息挖掘的首要任务。该类信息内容覆盖科研、生活、工作、娱乐等人类活动的各个方面，形式包含新闻、博客、论坛、微博、对话设计、学术期刊、商业单证等，可以满足文本挖掘等技术建模研究的需要。  
常用的数据获取收集方式有：
- 手动收集、下载
- 通过API：Application Programming Interface
- 数据爬虫

## 二、当前技术进展（国内外研究现状）
收集数据整个流程是一条流水线pipeline，并遵循以下步骤：
- 数据查找，定位
- 数据查询  
  - Urllib[1]  
  - Scapy[2]  
  - Selenium[3]  
  - …  
- 结果解析  
  - HTML解析: Beautiful Soup  
  - JSON  
  - XML  
  - 压缩文件GZ, Zip, Rar格式  
  - …  
- 本地存储  
  - 文件格式  
  - 数据库  
  - …  
- 数据使用

## 作业要求
- 准备一份热身作业  
- 收集，整理一份个人（文本）数据集，内容、体裁、格式、创意不限  
  - 参考暗恋日记  
- 将整理好的数据预处理，进行展示  
  - 例如示例中的词云展示  
- 加分/得分/扣分项：  
  - 有趣、好的创意、数据  
  - 生动的表达方式，图片、图表  
  - Matplotlib  
  - Jupyter notebook  
  - …  
- 正式的书写格式，参考毕业设计格式  
- 个人作业  
- 不允许抄袭、参考要列出出处，并作为文献参考  
- 提交日期：由班长和学习委员商量决定，可个人提交或三人以下组对提交

## 参考文献
1. URLlib: [https://docs.python.org/zh-cn/3.7/library/urllib.request.html](https://docs.python.org/zh-cn/3.7/library/urllib.request.html), Accessed 2021-03-15  
2. Scapy: [https://scapy.net/](https://scapy.net/), Accessed 2021-03-15  
3. Selenium: [https://www.selenium.dev/](https://www.selenium.dev/), Accessed 2021-03-15
