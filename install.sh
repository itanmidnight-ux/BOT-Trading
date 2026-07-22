#!/usr/bin/env bash
# install.sh — instala todas las dependencias del bot en Linux.
# Uso: ./install.sh
set -euo pipefail
cd "$(dirname "$0")"

BLUE='\033[1;34m'; GREEN='\033[1;32m'; RED='\033[1;31m'; NC='\033[0m'
info()  { echo -e "${BLUE}[install]${NC} $*"; }
ok()    { echo -e "${GREEN}[install]${NC} $*"; }
fail()  { echo -e "${RED}[install]${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Python >= 3.9 con venv disponible
# ---------------------------------------------------------------------------
command -v python3 >/dev/null 2>&1 || fail "python3 no esta instalado. En Debian/Ubuntu: sudo apt install python3 python3-venv python3-pip"

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
    || fail "Se requiere Python >= 3.9 (encontrado: $PYVER)"
info "Python $PYVER detectado"

python3 -m venv --help >/dev/null 2>&1 \
    || fail "El modulo venv no esta disponible. En Debian/Ubuntu: sudo apt install python3-venv"

# ---------------------------------------------------------------------------
# 2. Entorno virtual + dependencias Python
# ---------------------------------------------------------------------------
if [ ! -d .venv ]; then
    info "Creando entorno virtual en .venv/"
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
info "Actualizando pip e instalando dependencias de requirements.txt"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# ---------------------------------------------------------------------------
# 3. Archivo .env (privado, nunca se commitea: esta en .gitignore)
# ---------------------------------------------------------------------------
if [ ! -f .env ]; then
    cp .env.example .env
    chmod 600 .env
    ok "Creado .env desde .env.example (permisos 600, solo tu usuario lo lee)"
    info "  -> Edita .env y completa MT5_LOGIN / MT5_PASSWORD / MT5_SERVER antes de operar"
else
    chmod 600 .env
    info ".env ya existe, no se toca (permisos asegurados a 600)"
fi

mkdir -p logs

# ---------------------------------------------------------------------------
# 4. Verificacion: la suite de tests debe pasar sin MT5 conectado
# ---------------------------------------------------------------------------
info "Ejecutando tests de verificacion..."
python -m pytest tests/ -q || fail "Los tests fallaron: la instalacion no quedo consistente"

ok "Instalacion completa y verificada."
echo
echo "Siguientes pasos:"
echo "  1. Edita .env con tus credenciales MT5 (nano .env)"
echo "  2. En Linux el terminal MT5 corre bajo Wine: instala Wine + MT5 y levanta"
echo "     el servidor mt5linux con el Python de Wine (ver README.md, seccion Instalacion)"
echo "  3. Lanza el bot con:  ./run.sh          (DRY_RUN por defecto, sin ordenes reales)"
echo "     o corre los tests: ./run.sh --test"
