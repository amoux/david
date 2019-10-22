
import tensorflow as tf


def train(data, model, args):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)

        # Add embedding tensorboard visualization. Need tensorflow version
        # >= 0.12.0RC0
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'rnnlm/embedding:0'
        embed.metadata_path = args.metadata
        projector.visualize_embeddings(writer, config)

        max_iter = args.n_epoch * \
            (data.total_len // args.seq_length) // args.batch_size
        for i in range(max_iter):
            learning_rate = args.learning_rate * \
                (args.decay_rate ** (i // args.decay_steps))
            x_batch, y_batch = data.next_batch()
            feed_dict = {model.input_data: x_batch,
                         model.target_data: y_batch, model.lr: learning_rate}
            train_loss, summary, _, _ = sess.run(
                [model.cost, model.merged_op,
                 model.last_state, model.train_op], feed_dict)

            if i % 10 == 0:
                writer.add_summary(summary, global_step=i)
                print('Step:{}/{}, training_loss:{:4f}'.format(
                    i, max_iter, train_loss))
            if i % 2000 == 0 or (i + 1) == max_iter:
                saver.save(sess, os.path.join(
                    args.log_dir, 'jokes_model.ckpt'), global_step=i)
