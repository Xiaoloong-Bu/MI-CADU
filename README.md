Pre-trained model download link: https://pan.baidu.com/s/1ubXi5Q_cW3il8lqVvPnu4Q?pwd=4d55 code: 4d55

Test using ir_Random_noise as an example:
Adaptive:
```shell
python test.py --control 1 --save_path "./results" --data_path "D:\DeepLearning\Dataset\EMS\EMS_dataset_val(My_quality)\ir_Random_noise" --model_path "./CADU-97796-model.pt"
```
Manual:
```shell
python test.py --control 0 --hu_guid 1 --save_path "./results" --data_path "D:\DeepLearning\Dataset\EMS\EMS_dataset_val(My_quality)\ir_Random_noise" --model_path "./CADU-97796-model.pt"
```
No guidance:
```shell
python test.py --control 2 --save_path "./results" --data_path "D:\DeepLearning\Dataset\EMS\EMS_dataset_val(My_quality)\ir_Random_noise" --model_path "./CADU-97796-model.pt"
```