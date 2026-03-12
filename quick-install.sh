#!/bin/bash
set -e

#========== 配置区 ==========#
# ↓↓↓ 替换为你的 GitHub 仓库原始文件地址 ↓↓↓
GITHUB_RAW_BASE="https://raw.githubusercontent.com/lijiaze123/vertexai/main"
INSTALL_DIR="/opt/vertex-panel"
CONTAINER_NAME="vertex-panel"
IMAGE_NAME="vertex-panel"
PORT=9000
#========== 配置区 ==========#

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

print_banner() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  ${BOLD}Vertex AI Gemini 渠道管理面板 安装脚本${NC}${CYAN}  ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
    echo ""
}

log_info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()    { echo -e "${CYAN}[STEP]${NC} $1"; }

check_root() {
    if [ "$(id -u)" -ne 0 ]; then
        log_error "请使用 root 用户或 sudo 运行此脚本"
        exit 1
    fi
}

install_docker() {
    if command -v docker &> /dev/null; then
        log_info "Docker 已安装: $(docker --version)"
        return
    fi
    log_step "正在安装 Docker ..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    log_info "Docker 安装完成"
}

download_files() {
    log_step "正在下载文件 ..."
    mkdir -p "${INSTALL_DIR}"

    local files=("Dockerfile" "requirements.txt" "vertex_channel_panel.py" "panel_template.html")
    for file in "${files[@]}"; do
        log_info "下载 ${file} ..."
        if ! curl -sSfL "${GITHUB_RAW_BASE}/${file}" -o "${INSTALL_DIR}/${file}"; then
            log_error "下载 ${file} 失败，请检查仓库地址是否正确"
            exit 1
        fi
    done

    log_info "文件下载完成"
}

init_data_dir() {
    log_step "初始化数据目录 ..."
    mkdir -p "${INSTALL_DIR}/data"

    if [ ! -f "${INSTALL_DIR}/data/vertex_channels.json" ]; then
        cat > "${INSTALL_DIR}/data/vertex_channels.json" <<'EOF'
{
    "channels": [],
    "api_keys": []
}
EOF
        log_info "已创建默认配置文件 vertex_channels.json"
    else
        log_info "配置文件已存在，跳过"
    fi
}

build_image() {
    log_step "正在构建 Docker 镜像 ..."
    cd "${INSTALL_DIR}"
    docker build -t "${IMAGE_NAME}" .
    log_info "镜像构建完成"
}

run_container() {
    log_step "正在启动容器 ..."

    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_warn "发现已有容器 ${CONTAINER_NAME}，正在停止并移除 ..."
        docker rm -f "${CONTAINER_NAME}" > /dev/null 2>&1
    fi

    docker run -d \
        --name "${CONTAINER_NAME}" \
        --restart unless-stopped \
        -p "${PORT}:9000" \
        -v "${INSTALL_DIR}/data:/app/data" \
        "${IMAGE_NAME}"

    log_info "容器启动成功"
}

print_result() {
    local IP
    IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    [ -z "$IP" ] && IP="<服务器IP>"

    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           ✅ 部署完成！                  ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  面板地址:  ${BOLD}http://${IP}:${PORT}/panel${NC}"
    echo -e "  配置目录:  ${BOLD}${INSTALL_DIR}/data/${NC}"
    echo -e "  容器名称:  ${BOLD}${CONTAINER_NAME}${NC}"
    echo ""
    echo -e "  常用命令:"
    echo -e "    查看日志:  ${CYAN}docker logs -f ${CONTAINER_NAME}${NC}"
    echo -e "    重启服务:  ${CYAN}docker restart ${CONTAINER_NAME}${NC}"
    echo -e "    停止服务:  ${CYAN}docker stop ${CONTAINER_NAME}${NC}"
    echo -e "    卸载服务:  ${CYAN}docker rm -f ${CONTAINER_NAME} && rm -rf ${INSTALL_DIR}${NC}"
    echo ""
}

main() {
    print_banner
    check_root
    install_docker
    download_files
    init_data_dir
    build_image
    run_container
    print_result
}

main
