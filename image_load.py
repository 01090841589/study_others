from PIL import Image #pip install pillow
source_image = ".\\dog.jpg"
target_image = ".\\dog_resize.jpg"
image = Image.open(source_image)
# resize 할 이미지 사이즈 
resize_image = image.resize((640, 640))
# 저장할 파일 Type : JPEG, PNG 등 
# 저장할 때 Quality 수준 : 보통 95 사용 
resize_image.save(target_image, "JPEG", quality=95 )