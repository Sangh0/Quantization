# Quantization

### What?  
Quantization은 float type을 가지는 parameter를 integer type으로 변환하는 과정을 의미한다  
보통 32bit float을 8bit integer로 변환한다  

### Why?  
위의 예시처럼 32bit가 8bit로 줄어들기 때문에 메모리 사용량 및 inference 속도가 감소하는 효과를 얻을 수 있다  
그러나, 정확도가 떨어지는 것은 불가피하다  
weight 파일의 사이즈 축소, 연산량 감소, 효율적인 하드웨어 사용이 Quantization의 주 목적이라고 볼 수 있다

### Condition before applying Quantization
1. Inference Only  
    - quantization은 inference할 때만 사용  
    - 학습 시간을 줄이는 것과 관련 없음  

2. Not every layer can be quantized  
    - 딥러닝 모델의 모든 layer가 quantization이 될 수 없음  

3. Not every layer should be quantized  
    - 딥러닝 모델의 모든 layer가 quantization이 되어야 하는 것이 효율적인 것이 아님  

4. Not every model reacts the same way to quantization  
    - 같은 quantization을 적용해도 모든 딥러닝 모델이 동일한 효과를 내는 것은 아님  

5. Most available implementations are CPU only  
    - quantization이 적용되어도 CPU에서 연산해야 하는 경우도 존재함


#### 나중에 딥러닝 모델의 weight 파일을 Quantization 해야 할 일이 있을 때 보기 위해 만든 Repository
Reference : https://gaussian37.github.io/dl-concept-quantization/