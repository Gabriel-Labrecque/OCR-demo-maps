processed_img = prepare_image(image_data)

from app.celery_app import celery_app

print("Sending to MapReader...")

# .get() makes this synchronous/blocking
task = celery_app.send_task(
    "mapreader.extract_text",
    args=[processed_img],
    queue="mapreader_queue"
)
text_results = task.get(timeout=120)  # Wait up to 2 mins

# 3. POST-PROCESSING (Your existing code)
final_output = format_results(text_results)
return final_output



# INSTALL setup.sh
wget https://raw.githubusercontent.com/maps-as-data/MapReader/refs/heads/main/container/Dockerfile
wget https://raw.githubusercontent.com/maps-as-data/MapReader/refs/heads/main/container/requirements.txt
wget https://raw.githubusercontent.com/maps-as-data/MapReader/refs/heads/main/text-requirements.txt
