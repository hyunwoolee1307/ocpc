# 대기 변수 분석 환경 설정 지침 (해양기후예측용)

이 문서는 해양기후예측(OCP) 관점에서 대기 변수 분석을 안정적·재현가능하게 수행하기 위한 환경 설정 가이드입니다.

## 1) 권장 디렉토리 구조

```
data/      # 원본 관측/재분석/모델 자료
proc/      # 전처리 산출물
clim/      # 기후평년(월/계절)
fig/       # 시각화 결과
scripts/   # 분석 스크립트
notebooks/ # 탐색/검증용 노트북
env/       # 환경 설정 파일
```

## 2) 선호 도구(프로젝트 표준)

- **통계/탐색 분석**: R  
- **수치 계산(특히 선형 대수)**: Julia  
- **시각화**: Python  

특별한 기술적 이유(라이브러리 필요, 성능 제약 등)가 없는 한 위 기준을 우선합니다.

## 3) 파이썬 환경(Conda 권장)

### 권장 버전
- Python 3.11~3.13

### 필수/권장 패키지
- 데이터: `xarray`, `netCDF4`, `cfgrib`, `cftime`, `dask`
- 수치/분석: `numpy`, `scipy`, `pandas`
- 시각화: `matplotlib`, `cartopy`, `cmocean`
- 물리 진단: `windspharm`, `xgcm`, `xesmf`
- 유틸: `tqdm`, `rich`, `typer`

### 설치 예시
```
conda create -n ocpc_py313 python=3.13 -y
conda activate ocpc_py313
conda install -c conda-forge \
  xarray dask netcdf4 cfgrib cftime \
  numpy scipy pandas \
  matplotlib cartopy cmocean \
  windspharm xgcm xesmf \
  tqdm rich typer -y
```

## 3) 데이터 표준(권장)

### 좌표/변수 표준
- 위도/경도: `lat`, `lon`
- 시간: `time` (cftime)
- 수직: `level` 또는 `isobaricInhPa`

### 단위 통일
- 풍속: m/s
- 온도: K
- 기압: Pa 또는 hPa
- 강수: kg/m²/s 또는 mm/day

### 경도 체계
- 분석 내 일관성 유지 (0–360 또는 -180–180)

## 4) 대기 변수 기본 세트

필수
- 바람: `ugrd`, `vgrd`

권장
- 기압: `pres`/`mslp`
- 온도: `tmp`, `tmp2m`
- 습도: `q` 또는 `rh`
- 강수: `prate`
- 상층: 850/500/200 hPa

## 5) 기본 분석 파이프라인

1. 품질검사 (NaN, 범위, 단위)
2. 재격자/정렬 (공간·시간)
3. 기후평년 계산 (월/계절)
4. 이상치(anomaly) 계산
5. 진단 지표 계산 (ψ/χ, vorticity, divergence 등)
6. 시각화 (전지구 + 관심지역)

## 6) Helmholtz 분해 (ψ/χ)

권장: `windspharm` 사용

- ψ: rotational component
- χ: divergent component
- ψ 컨투어와 rotational wind 스트림라인이 평행하도록 계산

## 7) R/Julia (선택 사항)

필요에 따라 R 또는 Julia를 보조 도구로 사용할 수 있습니다. 이 경우에도 데이터 형식(NetCDF/GRIB)과 격자/좌표 표준을 일관되게 유지하세요.

### R (권장 패키지)
- `terra`, `stars`, `ncdf4`, `raster`, `sf`, `ggplot2`, `metR`, `fields`

### Julia (권장 패키지)
- `NCDatasets`, `NetCDF`, `ClimateBase`, `Interpolations`, `Plots`, `PyPlot`, `Cartopy`(PyCall 사용 시)

### 상호운용 가이드
- 교환 형식은 NetCDF 권장
- 좌표명은 `lat/lon/time/level`로 통일
- 단위 변환은 스크립트 또는 메타데이터에 명시

## 8) 재현성/운영 팁

- 입력/출력 경로를 config 파일로 분리
- 산출물 메타데이터에 기간/버전/격자 기록
- 결과 그림 파일명에 날짜/리드타임 포함
- 실행 로그 기록(입력/시간/버전)

## 8) 권장 파일 템플릿

- `env/ocpc.yml` (conda 환경)
- `scripts/preprocess.py`
- `scripts/compute_clim.py`
- `scripts/compute_anom.py`
- `scripts/plot_psi_chi.py`
