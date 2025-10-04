FROM golang:alpine AS builder

ENV GO111MODULE=on \
    CGO_ENABLED=0 \
    GOOS=linux

RUN apk add git

WORKDIR /build

# Download dependency
COPY . .
RUN go mod download


# Build
RUN go build -o main .

WORKDIR /dist

RUN cp /build/main .

FROM alpine:3.22.1 AS debug

COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /dist/main /

RUN apk add --no-cache bash curl bind-tools ca-certificates jq

ENTRYPOINT ["/main"]

# Use scratch image as finally tiny container 
FROM scratch AS prod

COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /dist/main /

ENTRYPOINT ["/main"]
