FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vertex_channel_panel.py .
COPY panel_template.html .

EXPOSE 9000

CMD ["python", "vertex_channel_panel.py"]
