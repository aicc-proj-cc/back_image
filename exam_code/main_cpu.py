# CPU서 동작

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
RABBITMQ_HOST = "localhost"  # RabbitMQ 서버의 실제 IP 또는 도메인
REQUEST_QUEUE = "image_generation_requests"
RESPONSE_QUEUE = "image_generation_responses"

# Stable Diffusion XL 모델 초기화 (CPU 환경)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float32  # CPU에서는 float32 사용
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "Linaqruf/animagine-xl-2.0",
    vae=vae,
    torch_dtype=torch.float32, # CPU에서는 float32 사용
    use_safetensors=True
    # variant="fp32"  # CPU 환경에 적합한 설정
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to("cpu") # 모델을 CPU로 이동
print("pipe :", pipe)

# 로라 불러올때 에러 해결
# pipe.load_lora_weights("Linaqruf/anime-detailer-xl-lora", weight_name="anime-detailer-xl.safetensors")
# pipe.fuse_lora(lora_scale=2)


# 이미지 저장 폴더 설정
OUTPUT_FOLDER = "generated_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # 폴더가 없으면 생성


def generate_image(data):
    """
    Stable Diffusion을 사용하여 이미지를 생성하고, Base64로 반환.
    """
    # prompt = data["prompt"]
    # negative_prompt = data.get("negative_prompt", "")

    # TODO : 프론트에서 정보 받아오도록 수정
    # 테스트용 프롬프트
    prompt = "face focus, cute, masterpiece, best quality, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck"
    negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    width = data.get("width", 256)
    height = data.get("height", 256)
    guidance_scale = data.get("guidance_scale", 7.5)
    num_inference_steps = data.get("num_inference_steps", 60)

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
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=RABBITMQ_HOST, heartbeat=6000)
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