FROM openjdk:8-jdk-alpine

RUN apk update && \
	apk add --no-cache libc6-compat ca-certificates && \
	ln -s /lib/libc.musl-x86_64.so.1 /lib/ld-linux-x86-64.so.2 && \
	rm -rf /var/cache/apk/*

ADD myVolume/ /Models

ADD TestDataset.csv /Data/TestDataset.csv

ADD prediction-app/ prediction-app

ENTRYPOINT [ "./prediction-app/bin/prediction-app" ]

# CMD ["/bin/sh"]