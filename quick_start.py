import os
import tensorflow as  tf

FLAGS= tf.app.flags.FLAGS


tf.app.flags.DEFINE_boolean("test_mode",False,"whether it is test mode. if it is test mode, only small percentage of data will be used")
tf.app.flags.DEFINE_string("data_path","./data/","path of traning data.")
tf.app.flags.DEFINE_string("mask_lm_source_file","./data/bert_train2.txt","path of traning data.")
tf.app.flags.DEFINE_string("ckpt_dir","./checkpoint_lm/","checkpoint location for the model")  # save to here, so make it easy to upload for test
tf.app.flags.DEFINE_integer("vocab_size",60000,"maximum vocab size.")
tf.app.flags.DEFINE_integer("d_model", 64, "dimension of model")  # 512--> 128
tf.app.flags.DEFINE_integer("num_layer", 6, "number of layer")
tf.app.flags.DEFINE_integer("num_header", 8, "number of header")
tf.app.flags.DEFINE_integer("d_k", 8, "dimension of k")  # 64
tf.app.flags.DEFINE_integer("d_v", 8, "dimension of v")  # 64

tf.app.flags.DEFINE_string("tokenize_style","word","checkpoint location for the model")
tf.app.flags.DEFINE_integer("max_allow_sentence_length",10,"max length of allowed sentence for masked language model")
tf.app.flags.DEFINE_float("learning_rate",0.0001,"learning rate") # 0.001
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
# decay_rate : 1， 表示学习率不改变
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.9, "percentage to keep when using dropout.")
tf.app.flags.DEFINE_integer("sequence_length",200,"max sentence length")#400
tf.app.flags.DEFINE_integer("sequence_length_lm",10,"max sentence length for masked language model")
tf.app.flags.DEFINE_boolean("is_training",True,"is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_boolean("is_fine_tuning",False,"is_finetuning.ture:this is fine-tuning stage")
tf.app.flags.DEFINE_integer("num_epochs",30,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_boolean("use_pretrained_embedding",False,"whether to use embedding or not.")#
tf.app.flags.DEFINE_string("word2vec_model_path","./data/Tencent_AILab_ChineseEmbedding_100w.txt","word2vec's vocabulary and vectors") # data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5--->data/news_12g_baidubaike_20g_novel_90g_embedding_64.bin--->sgns.merge.char
tf.app.flags.DEFINE_integer("process_num",35,"number of cpu process")


def main():

    configProto = tf.ConfigProto()
    configProto.gpu_options.allow_growth = True

    # 1. 创建session
    with tf.Session(config=configProto) as sess:
        # 2. 初始化模型类

        # 3. 初始化saver
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+ "checkpoint"):
            print ("restore vars from checkpoint.")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print  ('Initalizing vars')
            sess.run(tf.global_variables_initializer())




if __name__ == '__main__':
    print ("tf version:",tf.__version__)
    tf.app.run()



