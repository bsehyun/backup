class BinaryFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.9, from_logits=False, reduction=tf.keras.losses.Reduction.AUTO):
        super(BinaryFocalLoss, self).__init__(reduction=reduction)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        # Binary Crossentropy Loss
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)
        y_pred = tf.nn.sigmoid(y_pred) if self.from_logits else y_pred

        # Compute focal loss
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)  # p_t
        focal_weight = tf.pow(1 - pt, self.gamma)  # (1 - p_t) ^ gamma
        focal_loss = self.alpha * focal_weight * bce

        return tf.reduce_mean(focal_loss)
