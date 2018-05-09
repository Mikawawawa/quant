##我也不知道该怎么解释的东西

这个策略的主要逻辑是用市场偏好确定选股范围，用lstm确定第二天需要跟踪的股票池，在分钟级策略中，用AR模型来快速确定出手与否，不过具体要做的东西应该还任重道远。

mindgo平台提供了tensorflow、numpy、pandas等常见库，tf的时间序列库包含了lstm和ar的功能，所以具体实现时，只需要参考google提供的示例代码即可，大部分关键参数我已经标注出来了，但是调参没有怎么调，选择什么数据范围、选择延时、预测期数都需要在回测中去实践

##文件目录
- 对应模型的代码
	- ar.py
	- lstm.py
- 参考图像
	- ar_result.jpg
	- lstm_result.jpg
- 数据源
	- growth.csv（成长股股指）
	- multivariate_periods.csv(google示例用)
- 示例代码
	- lstm_g.py（google官方示例）

##数据源
这里的数据源是从csv文件中获取的，我用了两种读法
>     # 从csv中读取数据
    csv_file_name = './grove.csv'
    reader = tf.contrib.timeseries.CSVReader(csv_file_name)
	#ar.py

-----------------

>     # 通过numpy的array
    csv_file_name = './growth.csv'
    csv = pd.read_csv(csv_file_name, names=['date', 'radios'])
    _temp = csv.sort_values('date')
    data = {
        tf.contrib.timeseries.TrainEvalFeatures.TIMES: _temp['date'].as_matrix(),
        tf.contrib.timeseries.TrainEvalFeatures.VALUES: _temp['radios'].as_matrix(
        )
    }
    reader = NumpyReader(data)

mindgo自有的数据接口不知道返回什么，但是第二种比较通用，总能绕道numpy的array吧
第二种是先用pandas读取csv形成dataframe，然后取一个series，转换成array

##具体代码
我是谁，我在哪里，Google里的大神都是人才...

##输出
lstm部分
```
	observed_times = evaluation["times"][0]
    observed = evaluation["observed"][0, :, :]
    evaluated_times = evaluation["times"][0]
    evaluated = evaluation["mean"][0]
    predicted_times = predictions['times']
    predicted = predictions["mean"]
```
ar部分
```
    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)

    evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)

    # 这里的step是预测期数
    (predictions,) = tuple(ar.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation, steps=10)))
```
