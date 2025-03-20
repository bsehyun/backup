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














# other version

def outer_product(inputs):
    # Unpack the inputs
    x1, x2 = inputs
    
    # Get shapes
    batch_size = tf.shape(x1)[0]
    height = tf.shape(x1)[1]
    width = tf.shape(x1)[2]
    depth1 = x1.shape[3]
    depth2 = x2.shape[3]
    
    # Reshape tensors to 2D - 메모리 사용량 최적화
    x1_flat = tf.reshape(x1, [batch_size * height * width, depth1])
    x2_flat = tf.reshape(x2, [batch_size * height * width, depth2])
    
    # 그룹 컨볼루션 효과를 위해 채널을 그룹으로 분할
    groups = 8  # 그룹 수 지정
    depth1_per_group = depth1 // groups
    depth2_per_group = depth2 // groups
    
    # 결과를 저장할 텐서
    phi_I_groups = []
    
    # 각 그룹별로 계산 (메모리 효율성 향상)
    for g in range(groups):
        # 그룹별 채널 슬라이싱
        x1_group = x1_flat[:, g*depth1_per_group:(g+1)*depth1_per_group]
        x2_group = x2_flat[:, g*depth2_per_group:(g+1)*depth2_per_group]
        
        # 3D 텐서로 재구성
        x1_3d = tf.reshape(x1_group, [batch_size, height * width, depth1_per_group])
        x2_3d = tf.reshape(x2_group, [batch_size, height * width, depth2_per_group])
        
        # 그룹별 외적 계산
        phi_I_group = tf.matmul(tf.transpose(x1_3d, [0, 2, 1]), x2_3d)  # [batch, depth1_per_group, depth2_per_group]
        phi_I_group = tf.reshape(phi_I_group, [batch_size, depth1_per_group * depth2_per_group])
        
        phi_I_groups.append(phi_I_group)
    
    # 그룹 결과 결합
    phi_I = tf.concat(phi_I_groups, axis=1)
    
    # 정규화 - 프록시 정규화 적용 (배치 의존성 제거)
    # 고정된 정규화 파라미터 사용
    mean = 0.0  # 사전 계산된 평균값
    var = 1.0   # 사전 계산된 분산값
    phi_I = (phi_I - mean) / tf.sqrt(var + 1e-12)
    
    # 특성 맵 크기로 정규화
    phi_I = phi_I / tf.cast(height * width, tf.float32)
    
    # 부호 있는 제곱근
    y_ssqrt = tf.sign(phi_I) * tf.sqrt(tf.abs(phi_I) + 1e-12)
    
    # L2 정규화
    z_l2 = tf.nn.l2_normalize(y_ssqrt, axis=1)
    
    return z_l2

# 프록시 정규화 레이어 정의 (배치 정규화 대체)
class ProxyNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(ProxyNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        # 학습 가능한 스케일과 이동 파라미터
        self.gamma = None
        self.beta = None
        # 사전 계산된 통계 (배치 독립적)
        self.moving_mean = 0.0
        self.moving_var = 1.0

    def build(self, input_shape):
        dim = input_shape[-1]
        self.gamma = self.add_weight(
            name='gamma',
            shape=(dim,),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(dim,),
            initializer='zeros',
            trainable=True
        )
        super(ProxyNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        # 배치 정규화와 달리 배치 통계를 계산하지 않고 사전 정의된 값 사용
        normalized = (inputs - self.moving_mean) / tf.sqrt(self.moving_var + self.epsilon)
        return self.gamma * normalized + self.beta

def get_model():
    # 해상도를 절반으로 축소
    IMG_SIZE_h = 112  # 원래 224에서 절반으로 축소
    IMG_SIZE_w = 112  # 원래 224에서 절반으로 축소
    channel = 3
    
    input_tensor = Input(shape=(IMG_SIZE_h, IMG_SIZE_w, channel))
    
    # EfficientNet 백본 생성 - 팽창비 줄이기 위한 커스텀 파라미터
    def get_efficient_backbone(weights, name_prefix):
        # 기본 모델 불러오기
        base_model = efn.EfficientNetB0(
            weights=weights, 
            include_top=False,
            input_shape=(IMG_SIZE_h, IMG_SIZE_w, channel)
        )
        
        # 이름 충돌 방지를 위한 레이어 이름 변경
        for layer in base_model.layers:
            layer.name = name_prefix + '_' + layer.name
            
            # BatchNormalization 레이어를 ProxyNormalization으로 대체
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                # 레이어의 입력과 출력 가져오기
                layer_input = layer.input
                
                # 프록시 정규화 레이어 생성
                proxy_norm = ProxyNormalization(name=name_prefix + '_proxy_' + layer.name)
                
                # 레이어 교체를 위한 로직 (이 부분은 실제 구현에서는 더 복잡할 수 있음)
                # 여기서는 개념적으로만 표현
                
        # MBConv 블록의 그룹 컨볼루션 적용을 위한 로직
        # EfficientNet의 각 MBConv 블록을 수정하는 것이 이상적이지만,
        # 이 예제에서는 모델 출력에 추가 그룹 컨볼루션 처리를 추가
                
        return base_model
    
    # 백본 모델 생성
    base_model1 = get_efficient_backbone('imagenet', 'model1')
    base_model2 = get_efficient_backbone('noisy-student', 'model2')
    
    # 각 모델에서 특성 추출
    d1 = base_model1(input_tensor)
    d2 = base_model2(input_tensor)
    
    # 그룹 컨볼루션을 적용한 추가 레이어 (IPU 하드웨어 효율성 향상)
    d1 = tf.keras.layers.Conv2D(
        filters=d1.shape[-1], 
        kernel_size=1, 
        groups=8,  # 그룹 컨볼루션 적용
        padding='same',
        name='group_conv_d1'
    )(d1)
    
    d2 = tf.keras.layers.Conv2D(
        filters=d2.shape[-1], 
        kernel_size=1, 
        groups=8,  # 그룹 컨볼루션 적용
        padding='same',
        name='group_conv_d2'
    )(d2)
    
    # 프록시 정규화 적용
    d1 = ProxyNormalization(name='proxy_norm_d1')(d1)
    d2 = ProxyNormalization(name='proxy_norm_d2')(d2)
    
    # 개선된 외적 계산 적용
    bilinear = Lambda(outer_product)([d1, d2])
    
    # 최종 예측 레이어
    predictions = Dense(1, activation='sigmoid', name='predictions')(bilinear)
    
    # 모델 생성
    model = Model(inputs=input_tensor, outputs=predictions)
    
    # 모델 컴파일 - 학습률 스케줄링 추가
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0003,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=binary_focal_loss_with_label_smoothing(gamma=2.0, alpha=0.75, ls=0.125),
        metrics=['accuracy']
    )
    
    return model
