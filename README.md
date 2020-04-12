

bert中文文本多元分类

数据来源cnews，可以通过百度云下载

链接：https://pan.baidu.com/s/1LzTidW_LrdYMokN---Nyag 提取码：zejw

bert中文预训练模型下载地址：

链接：https://pan.baidu.com/s/14JcQXIBSaWyY7bRWdJW7yg 提取码：mvtl

训练运行命令

python run_cnews_cls.py --task_name=cnews --do_train=true --do_eval=true --do_predict=false --data_dir=cnews --vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --output_dir=model

运行测试命令

python run_cnews_cls.py --task_name=cnews --do_train=false --do_eval=false --do_predict=true --data_dir=cnews --vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --output_dir=result

效果 eval_accuracy = 0.93386775 eval_loss = 0.33081177 global_step = 468 loss = 0.3427003

原文链接：https://blog.csdn.net/qq236237606/article/details/105453660
