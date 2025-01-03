# 내 컴퓨터에서 OpenVLA 예제 코드 실행해보기 

[openvla 공식 git repository](https://github.com/openvla/openvla)


![demo-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/a4118998-3aa6-4cc6-92cb-d4f838ea7e3a)




## github 오픈소스 코드 clone 

```
git clone https://github.com/openvla/openvla.git

cd openvla
```

## 환경 설치 

```

conda create -n openvla python=3.10 -y
conda activate openvla

conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y  # UPDATE ME!

pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first

pip install packaging ninja

ninja --version; echo $?  # Verify Ninja --> should return exit code "0"

pip install "flash-attn==2.5.5" --no-build-isolation

```

### **[중요]** 

```
pip install accelerate==1.1.1
```

accelerate 버전을 위와 같이 안맞추어주면 이후 다음과 같은 에러 메시지를 볼 수 있습니다. 

```
ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models. Please use the model as it is, since the model has already been set to the correct devices and casted to the correct `dtype`.
```




## 목차 

- 1. verify_openvla.py 코드 실행해보기 
- 2. delopy.py 코드 실행해보기 
- 3. gradio로 시각화 
- 4. Libero 데이터셋으로 훈련시킨 OpenVLA 모델 Libero 로봇으로 eval 해보기 
- 5. ipynb으로 시나리오 별 시각화 결과 확인하기 
    - 1. LIBERO 데이터 확인하기 
    - 2. OpenVLA의 LIBERO 데이터에서 잘 작동하는지 확인하기 

## 1. 제공해준 verify_openvla.py 코드 실행해보기 

<code>vla-scripts/extern/verify_openvla.py</code> 파일이 있습니다.

가장 먼저 <code>import BitsAndBytesConfig</code>를 추가해주어야합니다. 

```
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
```

그리고 현재 가상환경에 아래 라이브러리를 설치해줍니다.

```
pip install bitsandbytes
```


좀만 내리면 <code>verify_openvla()</code> 함수가 보입니다.

제 컴퓨터로 현재 코드가 실행되지 않습니다.

8-bit 양자화 모드도 실행되지 않습니다. 

4-bit 양자화 모드는 됩니다. 따라서 4-bit 양자화 모드 주석을 해제합니다. 

```
# vla = AutoModelForVision2Seq.from_pretrained(
#     MODEL_PATH,
#     attn_implementation="flash_attention_2",
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
# ).to(device)

# === 8-BIT QUANTIZATION MODE (`pip install bitsandbytes`) :: [~9GB of VRAM Passive || 10GB of VRAM Active] ===
# print("[*] Loading in 8-Bit Quantization Mode")
# vla = AutoModelForVision2Seq.from_pretrained(
#     MODEL_PATH,
#     attn_implementation="flash_attention_2",
#     torch_dtype=torch.float16,
#     quantization_config=BitsAndBytesConfig(load_in_8bit=True),
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
# )

# === 4-BIT QUANTIZATION MODE (`pip install bitsandbytes`) :: [~6GB of VRAM Passive || 7GB of VRAM Active] ===
print("[*] Loading in 4-Bit Quantization Mode")
vla = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
```

다음으로 for문이 보입니다.

inputs 역시 양자화 모드로 바꿔주어야 합니다. 

```
print("[*] Iterating with Randomly Generated Images")
for _ in range(100):
    prompt = get_openvla_prompt(INSTRUCTION)
    image = Image.fromarray(np.asarray(np.random.rand(256, 256, 3) * 255, dtype=np.uint8))

    # === BFLOAT16 MODE ===
    # inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    # === 8-BIT/4-BIT QUANTIZATION MODE ===
    inputs = processor(prompt, image).to(device, dtype=torch.float16)

    # Run OpenVLA Inference
    start_time = time.time()
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    print(f"\t=>> Time: {time.time() - start_time:.4f} || Action: {action}")
```

실행해봅니다.

action 값이 나오는 것을 확인할 수 있습니다.


## 2. 제공해준 delopy.py 코드 실행해보기 

역시 요즘 인공지능 모델은 fastapi 배포 코드를 제공해주는 곳이 많은것 같다.

아마 huggingface에서 gradio 스타일로 많이 demo를 누구나 실행해 볼 수 있게 해서 그런것 같다. 

openvla 역시 <code>vla-scripts/deploy.py</code> 파일 코드를 살펴보면 fastapi가 있다. 

아래 라이브러리를 현재 가상환경에 설치하자 

```
pip install fastapi uvicorn 
```

역시 BitsAndBytesConfig 라이브러리를 import 해주어야한다. 

```
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
```

조금 내리면 Server Interface에 OpenVLAServer 클래스가 보인다. 

이 역시 현재 self.vla는 실행을 할 수 없다. 그래서 4비트 양자화 모드로 바꿔준다. 

```
# self.vla = AutoModelForVision2Seq.from_pretrained(
#     self.openvla_path,
#     attn_implementation=attn_implementation,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
# ).to(self.device)

print("[*] Loading in 4-Bit Quantization Mode")
self.vla = AutoModelForVision2Seq.from_pretrained(
    self.openvla_path,
    attn_implementation=attn_implementation,
    torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

```

조금더 내리면 predict_action 함수가 보입니다.

이 역시 inputs를 양자화 모드로 바꿔주어야 합니다. 

```
# inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)

# === 8-BIT/4-BIT QUANTIZATION MODE ===
inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.float16)
```

그러면 이제 터미널에서 <code>deploy.py</code> 를 실행합니다.

```
python vla-scripts/deploy.py
```

오류 없이 실행되는 것을 보실 수 있을 겁니다. 

## 3. gradio로 시각화 

자 그러면 이제 해당코드와 Interaction 할 수 있게 웹페이지로 시각화를 해봅시다.

먼저 root 위치에 images 폴더를 만들고 images 폴더에 example1.jpeg와 example2.jpeg 이미지를 넣어줍니다.

root 위치에 <code>webtest.py</code> 파일을 만들어 줍니다.

아래 코드를 복사 붙여넣기 해줍니다.

```
import gradio as gr
import requests
import json_numpy
import numpy as np
from PIL import Image

# Gradio 클라이언트와 서버 간 데이터 포맷 처리
json_numpy.patch()

# REST API 서버 엔드포인트
API_URL = "http://localhost:8000/act"

def predict_action(image, instruction, unnorm_key=None):
    # 업로드된 이미지를 numpy 배열로 변환
    image_array = np.array(image)

    # 요청 데이터(payload) 생성
    payload = {
        "image": image_array,
        "instruction": instruction,
    }

    if unnorm_key:
        payload["unnorm_key"] = unnorm_key

    # 서버에 POST 요청
    response = requests.post(API_URL, json=payload)
    
    # 서버 응답 확인
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error {response.status_code}: {response.text}"

# Gradio 인터페이스 구성
with gr.Blocks() as demo:
    gr.Markdown("# OpenVLA Robot Action Prediction")
    gr.Markdown(
        "Provide an image of the robot workspace and an instruction to predict the robot's action. "
        "You can either upload an image or provide a base64-encoded image via API."
    )

    with gr.Row():
        with gr.Column(scale=1):
            instruction_input = gr.Textbox(label="Instruction", placeholder="e.g., Pick up the remote")
            unnorm_key_input = gr.Textbox(label="Unnorm Key (Optional)", placeholder="e.g., bridge_orig")
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Submit")

        with gr.Column(scale=1):
            output_action = gr.Textbox(label="Robot Action (X, Y, Z, Roll, Pitch, Yaw)", interactive=False, lines=8)
    

    # 예측 함수 연결
    submit_btn.click(
        fn=predict_action,
        inputs=[image_input, instruction_input, unnorm_key_input],
        outputs=[output_action]
    )

    # 예제 제공
    gr.Examples(
        examples=[
            ["Place the red vegetable in the silver pot.", "bridge_orig", "./images/example1.jpeg"],
            ["Pick up the remote", "bridge_orig", "./images/example2.jpeg"]
        ],
        inputs=[instruction_input, unnorm_key_input, image_input]
    )

demo.launch()
```

아마 gradio를 설치하라는 오류 메시지가 나올 수 있습니다. 

```
pip install gradio
```

VScode에서 터미널 하나 더 열고 <code>webtest.py</code>를 실행해줍니다. 

language instruction과 image를 넣으면 action 값이 output 되는것을 확인할 수 있습니다. 

## 4. Libero 데이터셋으로 훈련시킨 OpenVLA 모델 Libero 로봇으로 eval 해보기 

<code>experiments/robot/libero</code> 폴더에 <code>run_libero_eval.py</code> 파일이 있습니다. 

조금 내려주시면 GenerateConfig 클래스가 있는데 

<code>load_in_4bit: bool = False</code>로 되어있다.

저는 해당 코드를 돌리기 위해서 True로 바꾸어줍니다. 

그리고 아래 openvla github repository README.md와 같이 코드를 실행하면 됩니다. 

```
# Launch LIBERO-Spatial evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True

# Launch LIBERO-Object evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True

# Launch LIBERO-Goal evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True

# Launch LIBERO-10 (LIBERO-Long) evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True
```

근데 이것들을 돌리면 process bar만 채워지고 어떻게 되어가는지 확인할 수 없습니다.

따라서 우리는 이제 ipynb을 통해서 하나의 사니리오씩 넣어주고 확인하고를 할 것입니다.

## 5. ipynb으로 시나리오 별 시각화 결과 확인하기 

### 1. LIBERO 데이터 확인하기 

먼저 우리는 LIBERO 데이터셋으로 학습된 OpenVLA를 LIBERO 데이터에 적용 실행하기 위해서 LIBERO 환경을 불러오고 시각화 할 것입니다. 

<code>**중요**</code> 여기서 가장 오래 시간을 버린 부분입니다. 

OpenVLA의 수행을 실제 데이터셋에 적용하고 실행하는 것을 보이기 위해 LIBERO를 선택했는데. 아무리 해도 오류가 발생하더군요. 해결방법은 

```
pip install robosuite==1.4
```

버전 문제였습니다. 

다음으로 아래 라이브러리도 설치해줍니다.

```
pip install --quiet mediapy
```

이제 제공한 01.test_LIBERO.ipynb을 모두 실행하고 확인해봅니다. 


### 2. OpenVLA의 LIBERO 데이터에서 잘 작동하는지 확인하기 

다음으로 제가 제공해드린 02.test_OpenVLA_on_LIBERO.ipynb을 모두 실행해보세요.


이제 여러분들이, CustomDataset을 구축하고 OpenVLA를 학습시켜 보세요.

그리고 저한테도 알려주세요.