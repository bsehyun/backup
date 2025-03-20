import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.keras.layers import Input, Dense, Lambda, GlobalAveragePooling2D, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def proxy_normalization(x):
    """ 배치 정규화 대신 프록시 정규화 적용 """
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    std = tf.math.reduce_std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + 1e-6)

def outer_product(inputs):
    x1, x2 = inputs
    batch_size, height, width, depth1 = tf.shape(x1)[0], tf.shape(x1)[1], tf.shape(x1)[2], x1.shape[3]
    depth2 = x2.shape[3]

    # Reshape tensors
    x1_flat = tf.reshape(x1, [batch_size * height * width, depth1])
    x2_flat = tf.reshape(x2, [batch_size * height * width, depth2])

    # Compute outer product
    phi_I = tf.matmul(tf.transpose(x1_flat, [1, 0]), x2_flat)  # [depth1, depth2]
    phi_I = tf.reshape(phi_I, [batch_size, depth1 * depth2])

    # Normalize
    phi_I = phi_I / tf.cast(height * width, tf.float32)
    y_ssqrt = tf.sign(phi_I) * tf.sqrt(tf.abs(phi_I) + 1e-12)
    z_l2 = tf.nn.l2_normalize(y_ssqrt, axis=1)

    return z_l2

def get_model():
    IMG_SIZE_h = 112  # 해상도를 절반으로 감소
    IMG_SIZE_w = 112
    channel = 3

    input_tensor = Input(shape=(IMG_SIZE_h, IMG_SIZE_w, channel))

    # EfficientNet 모델 생성
    base_model1 = efn.EfficientNetB0(weights='imagenet', include_top=False)
    base_model2 = efn.EfficientNetB0(weights='noisy-student', include_top=False)

    # 레이어 이름 변경 (충돌 방지)
    base_model1._name = "EfficientNetB0_imagenetWeight"
    base_model2._name = "EfficientNetB0_noisy-studentWeight"

    # Grouped Convolution 추가 (MBConv 최적화)
    def grouped_conv(x, groups=2):
        channels = x.shape[-1]
        group_size = channels // groups
        group_convs = []
        for i in range(groups):
            x_i = x[:, :, :, i * group_size: (i + 1) * group_size]
            conv_i = Conv2D(group_size, (3, 3), padding='same', use_bias=False)(x_i)
            conv_i = BatchNormalization()(conv_i)
            conv_i = ReLU()(conv_i)
            group_convs.append(conv_i)
        return tf.concat(group_convs, axis=-1)

    # EfficientNet 출력 가져오기
    d1 = base_model1(input_tensor)
    d2 = base_model2(input_tensor)

    # Proxy Normalization 적용
    d1 = Lambda(proxy_normalization)(d1)
    d2 = Lambda(proxy_normalization)(d2)

    # Grouped Convolution 적용
    d1 = grouped_conv(d1, groups=4)
    d2 = grouped_conv(d2, groups=4)

    # Bilinear Pooling 적용
    bilinear = Lambda(outer_product)([d1, d2])

    # 최종 예측 레이어
    predictions = Dense(1, activation='sigmoid', name='predictions')(bilinear)

    # 모델 생성
    model = Model(inputs=input_tensor, outputs=predictions)

    # 모델 컴파일
    model.compile(
        optimizer=Adam(learning_rate=0.0003, decay=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    return model
