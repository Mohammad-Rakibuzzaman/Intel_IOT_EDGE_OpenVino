FROM node:6.17.1

WORKDIR /app
# add `/app/node_modules/.bin` to $PATH
ENV PATH /app/node_modules/.bin:$PATH

COPY ["package.json", "webpack.config.js", "webpack.dev.config.js", "webpack.dist.config.js", "./"]

COPY ./ ./
RUN npm install -f && npm cache clean
EXPOSE 8080
CMD ["npm", "run", "dev"]