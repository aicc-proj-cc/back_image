from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL
import torch
import pika
import json
from io import BytesIO
from PIL import Image
import threading
import base64
import os

# RabbitMQ 호스트를 외부 IP로 설정
RABBITMQ_HOST = "222.112.27.120"  # RabbitMQ 서버의 실제 IP 또는 도메인
USERNAME = "guest"  # RabbitMQ 사용자 이름
PASSWORD = "guest"  # RabbitMQ 사용자 비밀번호
REQUEST_QUEUE = "image_generation_requests"
RESPONSE_QUEUE = "image_generation_responses"

# Stable Diffusion XL 모델 초기화 (GPU 환경)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16  # GPU에서는 float16 사용
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "Linaqruf/animagine-xl-2.0",
    vae=vae,
    torch_dtype=torch.float16,  # GPU에서는 float16 사용
    use_safetensors=True
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")  # 모델을 GPU로 이동
print("pipe :", pipe)

# 로라 가중치 경로 설정
lora_model_id = "Linaqruf/anime-detailer-xl-lora"  # LoRA 모델 경로
lora_weight_name = "anime-detailer-xl.safetensors"

try:
    # LoRA 가중치 로드
    pipe.load_lora_weights(lora_model_id, weight_name=lora_weight_name)
    print("LoRA 가중치 로드 완료")

    # LoRA 융합
    pipe.fuse_lora(lora_scale=2)
    print("LoRA 융합 완료")
except Exception as e:
    print(f"LoRA 적용 중 오류 발생: {e}")

# 이미지 저장 폴더 설정
OUTPUT_FOLDER = "generated_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # 폴더가 없으면 생성


def generate_image(data):
    """
    Stable Diffusion을 사용하여 이미지를 생성하고, Base64로 반환.
    """
    prompt = data.get("prompt", "default prompt")  # 사용자 프롬프트
    negative_prompt = data.get(
        "negative_prompt", "default negative prompt"
    )  # 네거티브 프롬프트
    width = data.get("width", 512)  # 기본값 512
    height = data.get("height", 512)  # 기본값 512
    guidance_scale = data.get("guidance_scale", 7.5)  # 기본값 7.5
    num_inference_steps = data.get("num_inference_steps", 50)  # 기본값 50

    # 이미지 생성
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]

    # 이미지를 Base64로 인코딩
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    return image, img_str


def async_save_image(image, image_filename):
    """
    비동기적으로 이미지를 저장합니다.
    """
    try:
        image.save(image_filename)
        print(f"이미지 저장 완료: {image_filename}")
    except Exception as e:
        print(f"이미지 저장 중 오류 발생: {e}")


def callback(ch, method, properties, body):
    """
    요청 큐에서 메시지를 소비하고, 이미지를 생성한 뒤 즉시 반환.
    추가 작업은 비동기적으로 처리.
    """
    data = json.loads(body)
    print(f"이미지 생성 요청 수신: {data['id']}")

    try:
        # 이미지 생성 및 Base64 인코딩
        image, img_str = generate_image(data)

        # 응답 전송: 프론트엔드에 즉시 반환
        response = {"id": data["id"], "image": img_str}
        ch.basic_publish(
            exchange="",
            routing_key=RESPONSE_QUEUE,
            body=json.dumps(response),
            properties=pika.BasicProperties(delivery_mode=1),
        )
        print(f"이미지 생성 완료 및 응답 전송: {data['id']}")

        # 비동기적으로 이미지 저장
        image_filename = os.path.join(OUTPUT_FOLDER, f"{data['id']}.png")
        threading.Thread(target=async_save_image, args=(image, image_filename)).start()

        # 메시지 확인 (ACK)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {e}")
        # 메시지 처리 실패 시 NACK 전송
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)


# RabbitMQ 연결 및 소비자 실행
credentials = pika.PlainCredentials(USERNAME, PASSWORD)
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=RABBITMQ_HOST, port=5672, credentials=credentials, heartbeat=6000)
)

channel = connection.channel()

channel.queue_declare(queue=REQUEST_QUEUE, durable=True)
channel.queue_declare(queue=RESPONSE_QUEUE, durable=True)

# prefetch_count 설정
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=REQUEST_QUEUE, on_message_callback=callback)

print("이미지 생성 서버가 실행 중입니다...")
channel.start_consuming()


# uvicorn main:app --reload --log-level debug --port 8002 