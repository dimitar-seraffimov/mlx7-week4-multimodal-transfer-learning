FROM python:3.10-slim

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  libpq-dev \
  wget \
  && rm -rf /var/lib/apt/lists/*

# install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the application
COPY . .

# expose the port that will be used by the streamlit app
EXPOSE 8080

# run the streamlit app
CMD ["streamlit", "run", "run_streamlit.py", "--server.port", "8080"]


