FROM node:
COPY . /app
WORKDIR /app
RUN pip install numpy 
EXPOSE 3000
