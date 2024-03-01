FROM node:6.17.1

WORKDIR /app
COPY ["package.json", "package-lock.json*", "./"]

RUN npm install -f && npm cache clean
RUN npm i jsonschema@1.2.6
COPY ./ ./
EXPOSE 8000
CMD [ "node","./server.js" ]