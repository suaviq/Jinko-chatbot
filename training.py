from enco_deco import *
from data_preparation import *

#----------------------- LOSS FUNCTION -------------------------------
class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self):
        self.name = 'masked_loss'
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    def __call__(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(y_true != 0, tf.float32)
        loss *= mask
        return tf.reduce_sum(loss)

#----------------------- TRAINING STEP -------------------------------
class TrainChatbot(tf.keras.Model):
    def __init__(self, embedding_dim, units, input_text_processor, output_text_processor, use_tf_function=True):
        super().__init__()
        encoder = Encoder(input_text_processor.vocabulary_size(), embedding_dim, units)
        decoder = Decoder(output_text_processor.vocabulary_size(), embedding_dim, units)
        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.use_tf_function = use_tf_function

    def train_step(self, inputs):
        if self.use_tf_function:
            return self._tf_train_step(inputs)
        else:
            return self._train_step(inputs)
      
    def _preprocess(self, input_text, target_text):
        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)
        input_mask = input_tokens != 0
        target_mask = target_tokens != 0
        return input_tokens, input_mask, target_tokens, target_mask

    def _train_step(self, inputs):
        input_text, target_text = inputs  
        (input_tokens, input_mask, target_tokens, target_mask) = self._preprocess(input_text, target_text)
        max_target_length = tf.shape(target_tokens)[1]

        with tf.GradientTape() as tape:
            enc_output, enc_state = self.encoder(input_tokens)
            dec_state = enc_state
            loss = tf.constant(0.0)

            for t in tf.range(max_target_length-1):
                new_tokens = target_tokens[:, t:t+2]
                step_loss, dec_state = self._loop_step(new_tokens, input_mask, enc_output, dec_state)
                loss = loss + step_loss
            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))
        variables = self.trainable_variables 
        gradients = tape.gradient(average_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return {'batch_loss': average_loss}

    def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):
        input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]
        decoder_input = DecoderInput(new_tokens=input_token, enc_output=enc_output, mask=input_mask)
        dec_result, dec_state = self.decoder(decoder_input, state=dec_state)
        y = target_token
        y_pred = dec_result.logits
        step_loss = self.loss(y, y_pred)

        return step_loss, dec_state
    
    @tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]), tf.TensorSpec(dtype=tf.string, shape=[None])]])
    def _tf_train_step(self, inputs):
        return self._train_step(inputs)

class BatchLogs(tf.keras.callbacks.Callback):
    def __init__(self, key):
        self.key = key
        self.logs = []

    def on_train_batch_end(self, n, logs):
        self.logs.append(logs[self.key])

def get_model():
    x, y = load_data('D:/chatbot_project/archive/dialogs.txt')
    input_text_processor, output_text_processor = text_vectorization(x, y)
    train_data, test_data, setup_data = tf_dataset(x, y)
    chatbot = TrainChatbot(embedding_dim, units,
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor,
    use_tf_function=False)
    chatbot.compile(
        optimizer=tf.optimizers.Adam(),
        loss=MaskedLoss(),
    )
    chatbot.use_tf_function = True
    batch_loss = BatchLogs('batch_loss')
    train_chatbot = TrainChatbot(embedding_dim, units, input_text_processor=input_text_processor, output_text_processor=output_text_processor)
    train_chatbot.compile(optimizer = tf.optimizers.Adam(), loss = MaskedLoss())
    return train_chatbot, train_data, batch_loss, input_text_processor, output_text_processor

def training():
    train_chatbot, train_data, batch_loss, input_text_processor, output_text_processor = get_model()
    train_chatbot.fit(train_data, epochs = 12, callbacks=[batch_loss])
    train_chatbot.save_weights('model/chatbot-trained')
    return train_chatbot, input_text_processor, output_text_processor

def main():
    training()

if __name__ == '__main__':
    main()
