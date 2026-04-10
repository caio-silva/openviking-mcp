VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")

.PHONY: build install clean test

build:
	go build -ldflags "-X main.version=$(VERSION)" -o openviking-mcp .

install:
	go install .

clean:
	rm -f openviking-mcp

test:
	go test ./... -race -count=1 -timeout 60s
