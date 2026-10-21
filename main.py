# -*- coding: utf-8 -*-
"""
===============================================================================
 TERMAL KUM BATARYASI (SAND BATTERY) - PARAMETRİK TASARIM VE SİMÜLASYON ARACI  (v4)
===============================================================================
 Çalıştırmak için:
     pip install streamlit pandas numpy plotly
     streamlit run main.py

 MODEL ÖZETİ
 -----------
 İki model katmanı birlikte çalışır:

 (1) 0-D TOPLU KAPASİTE MODELİ (hızlı, analitik/yarı-analitik)
     * Silo, yalıtımlı DİK SİLİNDİRİK bir kaptır (yarıçap R, yükseklik H).
     * Kum tek bir ısıl kütle gibi (tek sıcaklık) ele alınır.
     * c_p sabit ya da sıcaklığa bağımlı olabilir:  c_p(T) = c_p,25 + β (T − 25 °C)
     * Ana metrik kartları, soğuma eğrisi ve round-trip verim bu modelden gelir.

 (2) 1-D TERMOKLİN MODELİ (eksenel katmanlaşma, sonlu hacim)
     * Yatak, eksen boyunca N hücreye bölünür; ısı taşıyıcı akışkan (hava) yataktan
       geçer ve sıcaklık cephesi (termoklin) ilerler.
     * Akışkan–katı arasında yerel ısıl denge (LTE) varsayılır; eksenel iletim (k_eff),
       yan/üst/alt yüzey ısı kayıpları ve c_p(T) dahildir.
     * 0-D modelin "ideal karışık" üst sınırına karşı gerçekçi bir kapasite kullanımı
       ve verim tahmini verir.

 (3) v3 EKLERİ: parametre taraması (Analiz), optimizasyon (hedef enerji için optimum H/R
     ve yalıtım kalınlığı), JSON dışa/içe aktarma + silme, Markdown/TXT rapor çıktısı.

 (4) v4 EKLERİ: hibrit (kompozit) yatak malzemesi, CSP şarj kaynağı (Q_solar = A·DNI·η) ve
     N modüllü tesis (3D görselleştirme + yüzey alanına bağlı ısı kaybı trade-off analizi).
===============================================================================
"""

import inspect
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# =============================================================================
# 0) SABİTLER, MALZEME KÜTÜPHANESİ, VARSAYILAN KONFİGÜRASYON
# =============================================================================
J_PER_MWH = 3.6e9          # 1 MWh = 1e6 W * 3600 s = 3.6e9 J  (birim dönüşümü)
SEC_PER_HOUR = 3600.0
SEC_PER_DAY = 86400.0
DAYS_PER_MONTH = 30.44     # ortalama ay uzunluğu (365.25 / 12)
T_REF = 25.0               # c_p(T) ve entalpi için referans sıcaklık (°C)

ORANGE = "#f97316"
BLUE = "#3b82f6"
GREEN = "#10b981"
GRAY = "#94a3b8"
PROFILE_COLORS = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6", "#64748b"]

CP_CONST = "Sabit (ortalama c_p)"
CP_VAR = "Sıcaklığa bağımlı c_p(T)"
MIX_VOL = "Hacimce"
MIX_MASS = "Kütlece"
SRC_ELEC = "Elektrikli Rezistans"
SRC_CSP = "CSP (Fresnel Lens / Helyostat)"

# ρ: YIĞIN (bulk) yoğunluk. Boşluklu yatak için m = ρ_yığın·V kullanılır.
# c_p(T) = cp25 + β·(T − 25):  kuvars için 25 °C'de ~740, 500 °C'de ~1170 J/kg·K civarıdır
#   → β ≈ 0.9 J/kg·K². (Not: kuvarsın 573 °C'deki α→β faz geçişi bu lineer modelde yoktur.)
# 'cp' alanı sabit-c_p modu için 50–500 °C ortalamasıdır.
MATERIALS = {
    "Silika Kumu (Kuvars)": {"rho": 1600.0, "cp": 1000.0, "cp25": 740.0, "beta": 0.90},
    "Bazalt (kırma)": {"rho": 1750.0, "cp": 950.0, "cp25": 850.0, "beta": 0.42},
    # Refrakter (şamot) kırığı: yığın ≈ 1900 kg/m³, c_p(T) ≈ 840 + 0.35·(T−25) J/kg·K
    "Refrakter (şamot, kırma)": {"rho": 1900.0, "cp": 930.0, "cp25": 840.0, "beta": 0.35},
    "Manuel Giriş": {"rho": 1600.0, "cp": 1000.0, "cp25": 740.0, "beta": 0.90},
}
PRESET_NAMES = ["Silika Kumu (Kuvars)", "Bazalt (kırma)", "Refrakter (şamot, kırma)"]
CATEGORICAL = {
    "material": list(MATERIALS.keys()), "cp_mode": [CP_CONST, CP_VAR],
    "mat1": PRESET_NAMES, "mat2": PRESET_NAMES,
    "mix_basis": [MIX_VOL, MIX_MASS], "charge_src": [SRC_ELEC, SRC_CSP],
}

# Bir "konfigürasyon" = aşağıdaki anahtarların hepsi. Anahtar adları aynı zamanda
# yan menü widget'larının session_state anahtarlarıdır (yükleme/kaydetme bu sayede kolaydır).
DEFAULT_CFG = {
    "volume": 1000, "h_over_r": 3.0,
    "material": "Silika Kumu (Kuvars)", "rho": 1600.0,
    "cp_mode": CP_CONST, "cp": 1000.0, "cp25": 740.0, "beta": 0.90,
    "t_max": 550, "t_min": 50, "t_env": 15,
    "t_ins": 0.50, "k_ins": 0.060,
    "p_heat_mw": 5.0, "p_dis_mw": 2.0,
    "eta_heater_pct": 99.0, "h_out": 10.0, "hold_days": 2.0,
    "n_cells": 100, "k_eff": 0.8, "cut_ch_pct": 10, "cut_dis_pct": 70,
    # --- v4: hibrit karışım, şarj kaynağı, modüler tesis ---
    "hybrid": False, "mat1": "Refrakter (şamot, kırma)", "mat2": "Bazalt (kırma)",
    "mix_pct": 30, "mix_basis": MIX_VOL,
    "charge_src": SRC_ELEC, "csp_area": 10000.0, "csp_dni": 850.0, "csp_eta_pct": 70.0,
    "n_modules": 1,
}
CFG_KEYS = list(DEFAULT_CFG.keys())
CUR = "▶ Mevcut girdiler"

BAZALT_CFG = dict(DEFAULT_CFG, material="Bazalt (kırma)", rho=1750.0, cp=950.0, cp25=850.0, beta=0.42)

# Yan menü widget'larının izin verdiği aralıklar. JSON içe aktarmada değerler bu sınırlara
# kırpılır; aksi halde sınır dışı bir değer widget'ı hataya düşürürdü.
CFG_LIMITS = {
    "volume": (50, 20000), "h_over_r": (0.5, 8.0), "rho": (500.0, 5000.0),
    "cp": (400.0, 2000.0), "cp25": (400.0, 2000.0), "beta": (0.0, 3.0),
    "t_max": (200, 800), "t_min": (20, 200), "t_env": (-20, 45),
    "t_ins": (0.05, 2.0), "k_ins": (0.015, 1.0),
    "p_heat_mw": (0.1, 100.0), "p_dis_mw": (0.1, 100.0),
    "eta_heater_pct": (90.0, 100.0), "h_out": (3.0, 50.0), "hold_days": (0.0, 365.0),
    "n_cells": (20, 150), "k_eff": (0.1, 5.0), "cut_ch_pct": (1, 30), "cut_dis_pct": (30, 95),
    "mix_pct": (0, 100), "csp_area": (100.0, 1000000.0), "csp_dni": (100.0, 1400.0),
    "csp_eta_pct": (10.0, 95.0), "n_modules": (1, 20),
}


# =============================================================================
# 1) 0-D FİZİK FONKSİYONLARI (UI'dan bağımsız)
# =============================================================================
def silo_geometry(volume_m3, h_over_r):
    """
    V = π R² H ve H = (H/R)·R  →  R = [ V / (π · H/R) ]^(1/3).
    Hacim ile şekil birbirinden bağımsız ayarlanabilsin diye (V, H/R) girdi, (R, H) türetilir.
    (Kapalı silindirde yüzey alanını minimize eden oran H/R = 2'dir.)
    """
    r = (volume_m3 / (math.pi * h_over_r)) ** (1.0 / 3.0)
    return r, h_over_r * r


def model_params(cfg):
    """
    c_p modeli → (c0, β) çifti. Sabit modda β = 0 ve c0 = c_p; böylece tüm denklemler
    tek bir genel c_p(T) = c0 + β (T − 25) biçiminde yazılır (sabit hâl özel durumdur).
    """
    if cfg["cp_mode"] == CP_CONST:
        return float(cfg["cp"]), 0.0
    return float(cfg["cp25"]), float(cfg["beta"])


def cp_fun(T, c0, beta):
    """Özgül ısı [J/kg·K]:  c_p(T) = c0 + β (T − 25)."""
    return c0 + beta * (np.asarray(T, dtype=float) - T_REF)


def enthalpy(T, c0, beta):
    """
    Özgül (duyulur) entalpi [J/kg], c_p'nin sıcaklık üzerinden integrali:
        h(T) = ∫ c_p dT = c0 (T − 25) + (β/2) (T − 25)²
    Neden entalpi? c_p sıcaklıkla değiştiğinde Q = m c_p ΔT YANLIŞ olur; doğru ifade
    Q = m [h(T_max) − h(T_min)] dir.
    """
    x = np.asarray(T, dtype=float) - T_REF
    return c0 * x + 0.5 * beta * x * x


def stored_energy_mwh(volume, rho, c0, beta, t_max, t_min):
    """
    Duyulur ısı depolama:   Q = m [h(T_max) − h(T_min)],   m = ρ V
    Ortalama c_p = Q / (m ΔT) olarak ayrıca raporlanır (sabit modda c_p'ye eşittir).
    T_min altındaki ısı ısı değiştirici tarafından faydalı güce çevrilemez.
    """
    mass = rho * volume
    q_joule = mass * float(enthalpy(t_max, c0, beta) - enthalpy(t_min, c0, beta))
    cp_mean = q_joule / (mass * (t_max - t_min))
    return q_joule / J_PER_MWH, mass, cp_mean


def thermal_conductance(r, h, t_ins, k_ins, h_out):
    """
    Toplam ısıl iletkenlik UA [W/K] (paralel yollar; her yolda seri dirençler).
      Yan duvar (silindirik, radyal Fourier iletimi):
          R_yan = ln(r_o/r_i)/(2π k L) + 1/(h_d · A_d)
      Üst/alt kapak (düz levha):
          R_kapak = t/(k A) + 1/(h_d · A_d)
      UA = 1/R_yan + 2/R_kapak
    Dönüş: toplam UA, eşdeğer U, iç yüzey alanı, yan UA, TEK kapağın UA'sı
    (son ikisi 1-D modelde hücrelere dağıtılır).
    """
    r_o = r + t_ins
    R_side = math.log(r_o / r) / (2.0 * math.pi * k_ins * h) + 1.0 / (h_out * 2.0 * math.pi * r_o * h)
    R_end = t_ins / (k_ins * math.pi * r ** 2) + 1.0 / (h_out * math.pi * r_o ** 2)
    ua_side, ua_end = 1.0 / R_side, 1.0 / R_end
    ua = ua_side + 2.0 * ua_end
    a_inner = 2.0 * math.pi * r * h + 2.0 * math.pi * r ** 2
    return ua, ua / a_inner, a_inner, ua_side, ua_end


def ua_vec(r, h, t, k, h_out):
    """
    thermal_conductance() ile AYNI denklemlerin numpy-vektörel sürümü (UA, W/K).
    r, h, t dizi olabilir (broadcast) → optimizasyon ve tarama ızgaralarını döngüsüz hesaplar.
    """
    r_o = r + t
    R_side = np.log(r_o / r) / (2.0 * np.pi * k * h) + 1.0 / (h_out * 2.0 * np.pi * r_o * h)
    R_end = t / (k * np.pi * r ** 2) + 1.0 / (h_out * np.pi * r_o ** 2)
    return 1.0 / R_side + 2.0 / R_end


def composite_props(m1, m2, pct1, basis):
    """
    Hibrit (kompozit) yatak özellikleri — ağırlıklı ortalama.

    Oran HACİMCE verilirse (φ1 = pct1/100):
        ρ_karışım = φ1 ρ1 + φ2 ρ2                       (hacim ağırlıklı)
        w1 = φ1 ρ1 / ρ_karışım                          (kütle kesri)
    Oran KÜTLECE verilirse (w1 = pct1/100):
        1/ρ_karışım = w1/ρ1 + w2/ρ2                     (harmonik; hacimler toplanır)
        φ1 = w1 ρ_karışım / ρ1
    Özgül ısı HER ZAMAN kütle ağırlıklı ortalamadır:
        c_p = w1 c_p1 + w2 c_p2   (c_p(T) lineer olduğundan c_p,25 ve β da aynı şekilde karıştırılır)
    Neden? Isıl enerji kütleyle orantılıdır: ρ_k c_p,k = φ1 ρ1 c_p1 + φ2 ρ2 c_p2 özdeşliği tam sağlanır,
    yani karışımın hacimsel ısı kapasitesi iki bileşenin toplamıdır (ideal karışım: hacimler toplanır,
    boşluk doldurma etkisi ihmal edilir).
    """
    a, b = MATERIALS[m1], MATERIALS[m2]
    f1 = pct1 / 100.0
    if basis == MIX_VOL:
        phi1 = f1
        rho = phi1 * a["rho"] + (1.0 - phi1) * b["rho"]
        w1 = phi1 * a["rho"] / rho
    else:
        w1 = f1
        rho = 1.0 / (w1 / a["rho"] + (1.0 - w1) / b["rho"])
        phi1 = w1 * rho / a["rho"]
    w2 = 1.0 - w1
    return {"rho": rho, "cp": w1 * a["cp"] + w2 * b["cp"], "cp25": w1 * a["cp25"] + w2 * b["cp25"],
            "beta": w1 * a["beta"] + w2 * b["beta"], "phi1": phi1, "w1": w1}


def short_name(name):
    return name.split(" (")[0]


def material_label(cfg):
    """Ekranda/raporda kullanılan malzeme adı (hibritse karışım bileşimi)."""
    if cfg.get("hybrid"):
        pct = cfg["mix_pct"]
        return f"Hibrit %{pct} {short_name(cfg['mat1'])} + %{100 - pct} {short_name(cfg['mat2'])} ({cfg['mix_basis'].lower()})"
    return cfg["material"]


def csp_power_mw(cfg):
    """
    CSP şarj gücü:  Q_solar = A_optik · DNI · η_optik   [W → MW]
    A: toplam optik toplama alanı (m²), DNI: doğrudan normal ışınım (W/m²), η: optik verim (kesir).
    """
    return cfg["csp_area"] * cfg["csp_dni"] * cfg["csp_eta_pct"] / 100.0 / 1e6


def effective_cfg(cfg):
    """
    Şarj kaynağına göre 'etkin' konfigürasyon. CSP'de şarj gücü Q_solar olur ve zaten ısıl güç
    olduğundan ısıtıcı verimi uygulanmaz (η = 1). Böylece tüm model (0-D, 1-D, tarama,
    optimizasyon) değişmeden çalışır.
    """
    if cfg.get("charge_src") == SRC_CSP:
        return dict(cfg, p_heat_mw=csp_power_mw(cfg), eta_heater_pct=100.0)
    return cfg


# Gauss-Legendre düğümleri: düzgün integrantlar için az noktayla çok yüksek doğruluk.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(200)


def integrate(f, a, b):
    """∫_a^b f(x) dx  (Gauss-Legendre)."""
    xm, xr = 0.5 * (b + a), 0.5 * (b - a)
    return xr * float(np.sum(_GL_W * f(xm + xr * _GL_X)))


def build_cooling_model(mass, ua, t_env, t_max, c0, beta, s_max=40.0, n=4000):
    """
    Deşarjsız soğuma (Newton soğuma yasasının c_p(T)'li genellemesi).
        m c_p(T) dT/dt = −UA (T − T_çevre)
    θ = T − T_çevre ve s = ln(θ_max/θ) değişkenleriyle:
        t(s) = (m/UA) ∫_0^s c_p(T(s')) ds'
    c_p sabitse t = τ·s (klasik üstel çözüm). T(t) fonksiyonu bu (t, T) tablosunun
    interpolasyonuyla elde edilir; log-ızgara sayesinde T→T_çevre kuyruğu da doğru yakalanır.
    """
    s = np.linspace(0.0, s_max, n)
    T = t_env + (t_max - t_env) * np.exp(-s)
    cp = cp_fun(T, c0, beta)
    t = np.concatenate([[0.0], np.cumsum(0.5 * (cp[1:] + cp[:-1]) * np.diff(s))]) * mass / ua
    return {"s": s, "t": t, "T": T}


def cooling_series(cool, t_seconds, ua, t_env):
    """Verilen zaman dizisi için kum sıcaklığı ve ısı kaybı:  Q_kayıp = UA (T − T_çevre)."""
    temp = np.interp(t_seconds, cool["t"], cool["T"])
    return temp, ua * (temp - t_env) / 1000.0


def simulate_cycle(mass, c0, beta, ua, cool, t_env, t_min, t_max,
                   p_heat_w, eta_heater, p_dis_w, hold_s):
    """
    0-D şarj → bekleme → deşarj çevrimi. c_p(T) nedeniyle kapalı form yoktur; zamanı T
    üzerinden integre ederiz (m c_p dT = P_net dt − UA θ dt  →  dt = m c_p dT / (...)):

      Şarj   : t_ş = m ∫_{T_min}^{T_max}  c_p(T) / (η P_ısıtıcı − UA (T − T_çevre))  dT
      Bekleme: T_b = T(t_bekleme)  (soğuma modelinden)
      Deşarj : t_d = m ∫_{T_min}^{T_b}    c_p(T) / (P_deşarj + UA (T − T_çevre))    dT

    Verim:  η_RT = P_deşarj · t_d / (P_ısıtıcı · t_ş)     (elektrik → ısı → ısı)
    """
    p_net = eta_heater * p_heat_w
    if p_net <= ua * (t_max - t_env) * 1.000001:
        return None   # ısıtıcı T_max'taki kaybı bile karşılayamaz → hedefe ulaşılamaz
    cp = lambda T: cp_fun(T, c0, beta)
    t_charge = mass * integrate(lambda T: cp(T) / (p_net - ua * (T - t_env)), t_min, t_max)
    e_in = p_heat_w * t_charge / J_PER_MWH

    t_hold_end = float(np.interp(hold_s, cool["t"], cool["T"]))
    if t_hold_end <= t_min:
        t_dis = 0.0
    else:
        t_dis = mass * integrate(lambda T: cp(T) / (p_dis_w + ua * (T - t_env)), t_min, t_hold_end)
    e_out = p_dis_w * t_dis / J_PER_MWH
    return {
        "t_charge_h": t_charge / SEC_PER_HOUR, "t_discharge_h": t_dis / SEC_PER_HOUR,
        "t_after_hold": t_hold_end, "e_in_mwh": e_in, "e_out_mwh": e_out,
        "eta_rt": e_out / e_in if e_in > 0 else 0.0,
    }


def validate_cfg(cfg):
    """Fiziksel olarak anlamsız girdileri yakalar; sorun yoksa None döner."""
    if cfg["t_max"] <= cfg["t_min"]:
        return "T_max, T_min değerinden büyük olmalıdır."
    if cfg["t_min"] < cfg["t_env"]:
        return "T_min, T_çevre'den düşük olamaz (ısı pompası olmadan kum ortam altına soğutulamaz)."
    return None


def compute_system(cfg, name):
    """Bir konfigürasyonun tüm 0-D sonuçlarını tek sözlükte toplar (karşılaştırma için)."""
    c0, beta = model_params(cfg)
    # MODÜLER TESİS: toplam hacim N eşit siloya bölünür (aynı H/R ve yalıtım). Enerji (kütle) değişmez;
    # ısı kaybı yüzeylerden olduğu için toplam UA = N · UA_modül büyür:
    #     Q_kayıp = Σ U_i A_i (T − T_çevre) = N · UA(V/N) · (T − T_çevre)
    n_mod = max(1, int(cfg.get("n_modules", 1)))
    r, h = silo_geometry(cfg["volume"] / n_mod, cfg["h_over_r"])          # MODÜL başına geometri
    e_mwh, mass, cp_mean = stored_energy_mwh(cfg["volume"], cfg["rho"], c0, beta, cfg["t_max"], cfg["t_min"])
    ua_m, u_eff, a_m, _, _ = thermal_conductance(r, h, cfg["t_ins"], cfg["k_ins"], cfg["h_out"])
    ua, a_inner = n_mod * ua_m, n_mod * a_m
    # Referans: aynı toplam hacmin TEK dev silo olması (N = 1)
    r1, h1 = silo_geometry(cfg["volume"], cfg["h_over_r"])
    ua_single, _, a_single, _, _ = thermal_conductance(r1, h1, cfg["t_ins"], cfg["k_ins"], cfg["h_out"])
    mod_penalty_pct = (ua / ua_single - 1.0) * 100.0

    # Fourier ısı iletimi: Q_kayıp = U·A·(T_kum − T_çevre); şarj bitiminde T_kum = T_max → en büyük kayıp.
    q_loss_kw = ua * (cfg["t_max"] - cfg["t_env"]) / 1000.0
    t_dis_h = e_mwh / cfg["p_dis_mw"]                       # Q / P_deşarj (kayıpsız ideal)
    cool = build_cooling_model(mass, ua, cfg["t_env"], cfg["t_max"], c0, beta)
    tau_days = float(np.interp(1.0, cool["s"], cool["t"])) / SEC_PER_DAY   # e-katlanma süresi
    cycle = simulate_cycle(mass, c0, beta, ua, cool, cfg["t_env"], cfg["t_min"], cfg["t_max"],
                           cfg["p_heat_mw"] * 1e6, cfg["eta_heater_pct"] / 100.0,
                           cfg["p_dis_mw"] * 1e6, cfg["hold_days"] * SEC_PER_DAY)
    daily_loss_pct = q_loss_kw * 24.0 / (e_mwh * 1000.0) * 100.0
    eta_pct = cycle["eta_rt"] * 100.0 if cycle else float("nan")
    return {
        "name": name, "cfg": cfg, "c0": c0, "beta": beta, "r": r, "h": h, "mass": mass,
        "mass_t": mass / 1000.0, "cp_mean": cp_mean, "e_mwh": e_mwh, "ua": ua, "u_eff": u_eff,
        "a_inner": a_inner, "q_loss_kw": q_loss_kw, "t_dis_h": t_dis_h, "tau_days": tau_days,
        "cool": cool, "cycle": cycle, "eta_rt_pct": eta_pct, "daily_loss_pct": daily_loss_pct,
        "n_mod": n_mod, "v_mod": cfg["volume"] / n_mod, "ua_single": ua_single, "a_single": a_single,
        "mod_penalty_pct": mod_penalty_pct,
        "sub": {
            "e_mwh": f"{e_mwh * 1000 / cfg['volume']:,.0f} kWh/m³ · c̄p {cp_mean:,.0f}",
            "q_loss_kw": f"%{daily_loss_pct:.2f} / gün",
            "t_dis_h": f"≈ {t_dis_h / 24:,.1f} gün",
            "eta_rt_pct": f"Şarj {cycle['t_charge_h']:,.1f} sa" if cycle else None,
        },
    }


# =============================================================================
# 2) 1-D TERMOKLİN MODELİ
# =============================================================================
# Yönetici denklem (LTE adveksiyon–difüzyon, eksen z boyunca, taşıyıcı akış Ċ = ṁ·c_f):
#
#   ρ c_p(T) ∂T/∂t + (Ċ/A) ∂T/∂z = ∂/∂z ( k_eff ∂T/∂z ) − (UA_yan/(A·H)) (T − T∞)
#
#   • Sol terim   : yatağın ısıl depolaması (c_p sıcaklığa bağımlı)
#   • Adveksiyon  : akışkanın taşıdığı entalpi; sıcaklık cephesini (termoklin) ilerletir
#   • Difüzyon    : efektif eksenel iletim (kum + dağılma); termoklini kalınlaştırır
#   • Son terim   : yan duvardan kayıp; üst ve alt kapak kayıpları uç hücrelere eklenir
#
# Şarj: sıcak akışkan (T_in = T_max) ÜSTTEN girer, aşağı akar → sıcak üstte, soğuk altta.
#       Isıtıcı yükü modüle olur: P_ısıtıcı(t) = Ċ (T_max − T_çıkış)/η ≤ P_nominal.
# Deşarj: soğuk akışkan (T_in = T_min) ALTTAN girer, yukarı akar → kararlı katmanlaşma korunur.
# Yöntem: sonlu hacim, MUSCL (van Leer sınırlayıcı) + SSP-RK2 → sayısal difüzyonsuz, TVD.
#         Bekleme/soğuma fazlarında (adveksiyon yok) kararlılık kısıtından kurtulmak için
#         kapalı (backward-Euler) iletim çözümü kullanılır.
def _tc_setup(cfg, n_cells):
    c0, beta = model_params(cfg)
    r, h = silo_geometry(cfg["volume"], cfg["h_over_r"])
    _, _, _, ua_side, ua_end = thermal_conductance(r, h, cfg["t_ins"], cfg["k_ins"], cfg["h_out"])
    area = math.pi * r ** 2
    dz = h / n_cells
    ua_cell = np.full(n_cells, ua_side / n_cells)     # yan kayıp hücrelere eşit dağıtılır
    ua_cell[0] += ua_end                               # üst kapak kaybı üst hücreye
    ua_cell[-1] += ua_end                              # alt kapak kaybı alt hücreye
    return {"c0": c0, "beta": beta, "r": r, "h": h, "area": area, "dz": dz, "n": n_cells,
            "rho_A_dz": cfg["rho"] * area * dz, "kA_dz": cfg["k_eff"] * area / dz,
            "ua_cell": ua_cell, "mass": cfg["rho"] * area * h}


def _tc_rhs(T, Cdot, T_in, S, t_env):
    """dT/dt (her hücre) ve çıkış yüzeyi sıcaklığı. MUSCL: yüz değeri = T_i + ½ φ(r) ..."""
    N = T.size
    if Cdot > 0.0:
        Tp = np.empty(N + 1)
        Tp[0], Tp[1:] = T_in, T                        # hayalet hücre = giriş sıcaklığı
        a = Tp[1:] - Tp[:-1]                           # geri fark
        b = np.zeros(N)
        b[:-1] = T[1:] - T[:-1]                        # ileri fark (son hücrede 0 → çıkışta düz çıkarım)
        ab = a * b
        sig = np.where(ab > 0.0, 2.0 * ab / np.where(ab > 0.0, a + b, 1.0), 0.0)   # van Leer eğimi
        T_face = T + 0.5 * sig                         # i+1/2 yüzü (akış +i yönünde)
        T_face_in = np.empty(N)
        T_face_in[0], T_face_in[1:] = T_in, T_face[:-1]
        adv = Cdot * (T_face_in - T_face)              # akışkanla hücreye net giren ısı [W]
        t_out = T_face[-1]
    else:
        adv, t_out = 0.0, T[-1]
    cond = np.zeros(N)
    dT = T[1:] - T[:-1]
    cond[:-1] += S["kA_dz"] * dT                       # Fourier: komşular arası eksenel iletim
    cond[1:] -= S["kA_dz"] * dT
    loss = S["ua_cell"] * (T - t_env)
    return (adv + cond - loss) / (S["rho_A_dz"] * cp_fun(T, S["c0"], S["beta"])), t_out


def _tc_step(T, dt, Cdot, T_in, S, t_env):
    """SSP-RK2 (Heun): TVD özelliğini zamanda da korur."""
    k1, o1 = _tc_rhs(T, Cdot, T_in, S, t_env)
    k2, o2 = _tc_rhs(T + dt * k1, Cdot, T_in, S, t_env)
    return T + 0.5 * dt * (k1 + k2), 0.5 * (o1 + o2)


def _tc_dt(T, Cdot, S, cfl=0.4):
    """Adveksiyon CFL ≤ 0.4 ve iletim kararlılığı için adım büyüklüğü."""
    cmin = S["rho_A_dz"] * float(cp_fun(T.min(), S["c0"], S["beta"]))
    dt = cfl * cmin / Cdot if Cdot > 0.0 else np.inf
    return min(dt, 0.4 * cmin / (2.0 * S["kA_dz"]))


def _implicit_advance(T, duration, n_steps, S, t_env):
    """Akışsız (bekleme/soğuma) faz: kapalı iletim + yüzey kaybı, koşulsuz kararlı."""
    N = T.size
    off = -S["kA_dz"] * (np.eye(N, k=1) + np.eye(N, k=-1))
    nb = np.full(N, 2.0 * S["kA_dz"])
    nb[0] = nb[-1] = S["kA_dz"]
    dt = duration / n_steps
    losses = []
    for _ in range(n_steps):
        C = S["rho_A_dz"] * cp_fun(T, S["c0"], S["beta"])
        M = off.copy()
        M[np.diag_indices(N)] = C / dt + S["ua_cell"] + nb
        T = np.linalg.solve(M, C / dt * T + S["ua_cell"] * t_env)
        losses.append(float(np.sum(S["ua_cell"] * (T - t_env))))
    return T, losses


def _first_crossing_down(T, z, level):
    """z boyunca azalan T profilinin 'level' seviyesini ilk kez geçtiği konum (lineer interpolasyon)."""
    idx = np.where(T < level)[0]
    if idx.size == 0:
        return None
    i = idx[0]
    if i == 0:
        return float(z[0])
    return float(z[i - 1] + (level - T[i - 1]) / (T[i] - T[i - 1]) * (z[i] - z[i - 1]))


def thermocline_cycle(cfg):
    """Tam çevrim: şarj → bekleme → deşarj (1-D). Sonuçlar sözlük olarak döner."""
    n_mod = int(cfg.get("n_modules", 1))
    if n_mod > 1:
        # Özdeş modüller paralel çalışır: bir modülü (V/N, P/N) çöz, enerjileri N ile çarp.
        # Süreler, yüzde oranları ve profiller modül başınadır.
        cfg_m = dict(cfg, volume=cfg["volume"] / n_mod, p_heat_mw=cfg["p_heat_mw"] / n_mod,
                     p_dis_mw=cfg["p_dis_mw"] / n_mod, n_modules=1)
        res = thermocline_cycle(cfg_m)
        res["e_in_mwh"] *= n_mod
        res["e_out_mwh"] *= n_mod
        return res
    S = _tc_setup(cfg, int(cfg["n_cells"]))
    N, dz, h = S["n"], S["dz"], S["h"]
    t_env, t_min, t_max = cfg["t_env"], cfg["t_min"], cfg["t_max"]
    d_tot = t_max - t_min
    eta = cfg["eta_heater_pct"] / 100.0
    p_heat, p_dis = cfg["p_heat_mw"] * 1e6, cfg["p_dis_mw"] * 1e6
    # Akışkan kapasite hızı Ċ: nominal gücü soğuk yatakta (ΔT = T_max − T_min) taşıyacak şekilde seçilir.
    cdot_ch, cdot_dis = eta * p_heat / d_tot, p_dis / d_tot
    cut_ch = t_min + cfg["cut_ch_pct"] / 100.0 * d_tot       # şarj: çıkış bu değeri aşınca bitir
    cut_dis = t_min + cfg["cut_dis_pct"] / 100.0 * d_tot     # deşarj: çıkış bunun altına inince bitir
    q_nom_j = S["mass"] * float(enthalpy(t_max, S["c0"], S["beta"]) - enthalpy(t_min, S["c0"], S["beta"]))

    # ---- ŞARJ (üstten sıcak akışkan) ----
    T = np.full(N, float(t_min))
    t, e_in, guard = 0.0, 0.0, 0
    max_t = 6.0 * q_nom_j / (eta * p_heat) + 1.0
    while T[-1] < cut_ch and t < max_t and guard < 200000:
        dt = _tc_dt(T, cdot_ch, S)
        T, t_out = _tc_step(T, dt, cdot_ch, t_max, S, t_env)
        e_in += cdot_ch * (t_max - t_out) * dt / eta         # şebekeden çekilen elektrik [J]
        t += dt
        guard += 1
    t_charge = t
    prof_charge = T.copy()

    # ---- BEKLEME ----
    hold_s = cfg["hold_days"] * SEC_PER_DAY
    if hold_s > 0:
        T, _ = _implicit_advance(T, hold_s, int(np.clip(hold_s / 7200.0, 10, 400)), S, t_env)
    prof_hold = T.copy()

    # ---- DEŞARJ (alttan soğuk akışkan; akış koordinatında indeks 0 = alt = giriş) ----
    T = T[::-1].copy()
    t_d, e_out, guard = 0.0, 0.0, 0
    ts, touts, ps = [], [], []
    snaps, thr = [], [0.25, 0.5, 0.75]
    while T[-1] >= cut_dis and t_d < max_t and guard < 200000:
        dt = _tc_dt(T, cdot_dis, S)
        T, t_out = _tc_step(T, dt, cdot_dis, t_min, S, t_env)
        p_w = cdot_dis * (t_out - t_min)                     # kullanıcıya verilen ısıl güç
        e_out += p_w * dt
        t_d += dt
        guard += 1
        ts.append(t_d / SEC_PER_HOUR); touts.append(t_out); ps.append(p_w / 1000.0)
        while thr and e_out >= thr[0] * q_nom_j:
            snaps.append((f"%{int(thr[0] * 100)} deşarj", T[::-1].copy()))
            thr.pop(0)
    prof_end = T[::-1].copy()

    # Termoklin kalınlığı: %50 deşarj anında, ΔT'nin %90 → %10 seviyeleri arası mesafe
    z = (np.arange(N) + 0.5) * dz
    snap50 = next((p for lbl, p in snaps if lbl == "%50 deşarj"), None)
    thick = float("nan")
    if snap50 is not None:
        z_hi = _first_crossing_down(snap50, z, t_min + 0.9 * d_tot)
        z_lo = _first_crossing_down(snap50, z, t_min + 0.1 * d_tot)
        if z_hi is not None and z_lo is not None:
            thick = z_lo - z_hi

    e_in_mwh, e_out_mwh = e_in / J_PER_MWH, e_out / J_PER_MWH
    profiles = [("Şarj sonu", prof_charge), ("Bekleme sonu", prof_hold)] + snaps + [("Deşarj sonu", prof_end)]
    return {
        "t_charge_h": t_charge / SEC_PER_HOUR, "t_dis_h": t_d / SEC_PER_HOUR,
        "e_in_mwh": e_in_mwh, "e_out_mwh": e_out_mwh,
        "eta_pct": e_out_mwh / e_in_mwh * 100.0 if e_in_mwh > 0 else float("nan"),
        "util_pct": e_out / q_nom_j * 100.0, "thick_m": thick,
        "x_h": h - z, "profiles": profiles,
        "dis_t_h": np.array(ts), "dis_Tout": np.array(touts), "dis_P_kw": np.array(ps),
    }


def thermocline_cooling(cfg, horizon_s):
    """
    1-D soğuma: tam şarjlı (üniform T_max) yatağın deşarjsız soğuması.
    Uç hücrelerin (kapaklar) daha çok kaybetmesi ve eksenel iletim yakalanır.
    """
    n_mod = max(1, int(cfg.get("n_modules", 1)))
    S = _tc_setup(dict(cfg, volume=cfg["volume"] / n_mod), 40)             # tek modül; sonuç N ile çarpılır
    n_steps = int(np.clip(horizon_s / SEC_PER_DAY, 120, 800))
    T0 = np.full(S["n"], float(cfg["t_max"]))
    loss0 = float(np.sum(S["ua_cell"] * (T0 - cfg["t_env"])))
    _, losses = _implicit_advance(T0, horizon_s, n_steps, S, cfg["t_env"])
    t = np.linspace(0.0, horizon_s, n_steps + 1)
    return t, n_mod * np.array([loss0] + losses) / 1000.0


def freeze(cfg):
    """Konfigürasyonu önbellek anahtarı olarak kullanılabilecek hashlenebilir bir yapıya çevirir."""
    return tuple((k, cfg[k]) for k in CFG_KEYS)


@st.cache_data(show_spinner="1-D termoklin modeli çözülüyor…")
def tc_cycle_cached(frozen):
    return thermocline_cycle(dict(frozen))


@st.cache_data(show_spinner=False)
def tc_cooling_cached(frozen, horizon_months):
    return thermocline_cooling(dict(frozen), horizon_months * DAYS_PER_MONTH * SEC_PER_DAY)


# =============================================================================
# 3) UI YARDIMCILARI
# =============================================================================
def wide_call(fn, *args, **kwargs):
    """Streamlit sürüm uyumluluğu: yeni `width="stretch"`, eski `use_container_width=True`."""
    try:
        if "width" in inspect.signature(fn).parameters:
            return fn(*args, width="stretch", **kwargs)
    except Exception:
        pass
    return fn(*args, use_container_width=True, **kwargs)


def style_fig(fig, title, height=470):
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)), height=height, hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.25)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.25)", zeroline=False)
    return fig


def is_num(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


# Metrik kartı tanımları: (anahtar, etiket, biçim, birim, "better", yardım)
# better: "up"  → artış iyi  (yeşil ↑ / kırmızı ↓)     delta_color="normal"
#         "down"→ düşüş iyi  (yeşil ↓ / kırmızı ↑)     delta_color="inverse"
#         "neutral" → iyi/kötü yok (gri)               delta_color="off"
MAIN_SPECS = [
    ("e_mwh", "⚡ Depolanan Isıl Enerji", "{:,.1f}", "MWh", "up", "Q = m·[h(T_max) − h(T_min)],  h(T) = ∫ c_p dT"),
    ("q_loss_kw", "🔥 Isı Kaybı (T_max'ta)", "{:,.1f}", "kW", "down", "Q_kayıp = U·A·(T_kum − T_çevre); şarj bitiminde (en yüksek ΔT)."),
    ("t_dis_h", "⏱️ Maks. Deşarj Süresi", "{:,.1f}", "saat", "up", "Toplam kullanılabilir enerji / deşarj gücü (kayıpsız ideal)."),
    ("eta_rt_pct", "🔄 Round-trip Verim", "{:,.1f}", "%", "up", "Şarj girdisi (elektrik veya CSP'de toplanan güneş ısısı) → verilen ısı; şarj + bekleme + deşarj kayıpları dahil (0-D model)."),
]
DERIVED_SPECS = [
    ("r", "Silo yarıçapı", "{:,.2f}", "m", "neutral", "R = [V / (π·H/R)]^(1/3)"),
    ("h", "Silo yüksekliği", "{:,.2f}", "m", "neutral", "H = (H/R)·R"),
    ("mass_t", "Kum kütlesi", "{:,.0f}", "ton", "neutral", "m = ρ_yığın · V"),
    ("u_eff", "Eşdeğer U", "{:,.3f}", "W/m²K", "down", "U = UA / A_iç"),
    ("tau_days", "Zaman sabiti τ", "{:,.0f}", "gün", "up", "T − T_çevre farkının 1/e'ye düşme süresi."),
    ("a_inner", "İç yüzey alanı", "{:,.0f}", "m²", "neutral", "A = 2πRH + 2πR²"),
]
TC_SPECS = [
    ("util_pct", "Kapasite kullanımı", "{:,.1f}", "%", "up", "Çekilen ısı / nominal Q. Termoklin ve kayıplar yüzünden 100%'ün altındadır."),
    ("eta_pct", "1-D Round-trip verim", "{:,.1f}", "%", "up", "Gerçekçi (katmanlaşmış) yatakta elektrik → ısı verimi."),
    ("t_charge_h", "Şarj süresi", "{:,.1f}", "saat", "down", "Çıkış sıcaklığı şarj eşiğine ulaşana kadar."),
    ("t_dis_h", "Gerçek deşarj süresi", "{:,.1f}", "saat", "up", "Çıkış sıcaklığı deşarj eşiğinin altına inene kadar."),
    ("thick_m", "Termoklin kalınlığı", "{:,.2f}", "m", "down", "%50 deşarj anında ΔT'nin %90 ile %10 seviyeleri arasındaki mesafe."),
]


def render_cards(res, ref, specs, ncols):
    """
    st.metric kartlarını çizer. `ref` verilirse (karşılaştırma modunda B sistemi için)
    delta = değer − referans olarak hesaplanır ve mühendislik yönüne göre renklendirilir.
    """
    for start in range(0, len(specs), ncols):
        cols = st.columns(ncols)
        for col, (key, label, fmt, unit, better, hlp) in zip(cols, specs[start:start + ncols]):
            v = res.get(key)
            value = f"{fmt.format(v)} {unit}" if is_num(v) else "—"
            delta, dcolor = None, "off"
            if ref is None:
                delta = (res.get("sub") or {}).get(key)
            elif is_num(v) and is_num(ref.get(key)):
                d = v - ref[key]
                if float(fmt.format(d).replace(",", "")) != 0.0:
                    delta = f"{fmt.replace('{:', '{:+').format(d)} {unit}"
                    dcolor = {"up": "normal", "down": "inverse"}.get(better, "off")
            col.metric(label, value, delta=delta, delta_color=dcolor, help=hlp)


def md_table(rows):
    out = "| Sembol | Açıklama | Birim |\n|:--|:--|:--|\n"
    return out + "\n".join(f"| {s} | {d} | {u} |" for s, d, u in rows)


# =============================================================================
# 3b) ANALİZ, OPTİMİZASYON, JSON VE RAPOR FONKSİYONLARI
# =============================================================================
OBJ_LOSS = "Isı kaybını minimize et"
OBJ_COST = "Toplam maliyeti minimize et (yalıtım + ısı kaybı)"

OPT_SPECS = [
    ("volume", "Gerekli kum hacmi", "{:,.0f}", "m³", "neutral", "V = Q_hedef / (ρ·[h(T_max) − h(T_min)])"),
    ("hr", "Optimum H/R", "{:,.2f}", "–", "neutral", "Arama aralığı içindeki en iyi yükseklik/yarıçap oranı."),
    ("t_ins", "Optimum yalıtım kalınlığı", "{:,.2f}", "m", "neutral", "Arama aralığı içindeki en iyi yalıtım kalınlığı."),
    ("q_loss_kw", "Isı kaybı (T_max'ta)", "{:,.1f}", "kW", "down", "Δ = optimum − mevcut şekil (aynı enerji, aynı hacim)."),
    ("daily_loss_pct", "Günlük kayıp", "{:,.2f}", "%/gün", "down", "Başlangıç ısı kaybının depolanan enerjiye oranı."),
    ("eta_rt_pct", "Round-trip verim (0-D)", "{:,.1f}", "%", "up", "Optimum tasarımın 0-D round-trip verimi."),
]
OPT_COST_SPEC = ("cost_total", "Toplam maliyet", "{:,.0f}", "€", "down",
                 "Yalıtım malzemesi + ömür boyu ısı kaybının değeri.")


def sanitize_cfg(raw):
    """
    Dışarıdan gelen (JSON) bir konfigürasyonu güvenli hâle getirir: eksik anahtarlar varsayılandan
    doldurulur, tipler düzeltilir, sayılar widget sınırlarına kırpılır, bilinmeyen anahtarlar atılır.
    """
    cfg = dict(DEFAULT_CFG)
    for k in CFG_KEYS:
        if k not in raw:
            continue
        v, d = raw[k], DEFAULT_CFG[k]
        try:
            if k in CATEGORICAL:
                v = v if v in CATEGORICAL[k] else d
            elif k == "hybrid":
                v = bool(v)
            else:
                v = float(v)
                if not math.isfinite(v):
                    raise ValueError
                lo, hi = CFG_LIMITS[k]
                v = min(max(v, lo), hi)
                v = int(round(v)) if isinstance(d, int) else float(v)
        except (TypeError, ValueError):
            v = d
        cfg[k] = v
    if cfg["hybrid"]:   # hibritte ρ ve c_p daima bileşenlerden yeniden hesaplanır (tutarlılık)
        comp = composite_props(cfg["mat1"], cfg["mat2"], cfg["mix_pct"], cfg["mix_basis"])
        cfg.update({kk: comp[kk] for kk in ("rho", "cp", "cp25", "beta")})
    return cfg


def parse_import(raw_bytes):
    """JSON baytlarını {isim: konfigürasyon} sözlüğüne çevirir. Dönüş: (konfigürasyonlar, atlanan_sayısı)."""
    try:
        data = json.loads(raw_bytes.decode("utf-8-sig"))
    except Exception as exc:
        raise ValueError("Dosya geçerli bir JSON değil.") from exc
    raw = data.get("configs") if isinstance(data, dict) and "configs" in data else data
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Dosyada konfigürasyon bulunamadı.")
    out, skipped = {}, 0
    for name, c in raw.items():
        nm = str(name).strip()[:80]
        if not nm or not isinstance(c, dict):
            skipped += 1
            continue
        cfg = sanitize_cfg(c)
        if validate_cfg(cfg):          # fiziksel olarak anlamsız kayıtlar (örn. T_max ≤ T_min) alınmaz
            skipped += 1
            continue
        out[nm] = cfg
    if not out:
        raise ValueError("İçe aktarılabilir geçerli konfigürasyon bulunamadı.")
    return out, skipped


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def export_json(configs):
    """Kayıtlı tüm konfigürasyonları indirilebilir JSON baytlarına çevirir."""
    payload = {"format": "kum-bataryasi-konfigurasyonlari", "version": 3,
               "exported": datetime.now().isoformat(timespec="seconds"), "configs": configs}
    return json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default).encode("utf-8")


def design_objective(volume, hr, t, base, objective, prices):
    """
    Tasarım değişkenleri (H/R, yalıtım kalınlığı t) için amaç fonksiyonu. Hacim sabittir
    (hedef enerji → V = Q/(ρ Δh)). hr ve t dizi olabilir (broadcast).

      Amaç 1 (kayıp)  : f = UA (T_max − T_çevre)           [kW]
      Amaç 2 (maliyet): f = c_ins · V_ins + c_E · E_kayıp,yıl · N_yıl
          V_ins     = π (R+t)² (H+2t) − π R² H              (yalıtım malzemesi hacmi)
          E_kayıp,yıl = UA (T̄ − T_çevre) · 8760 h,  T̄ = (T_max + T_min)/2
    Neden iki amaç? Yalnızca ısı kaybı minimize edilirse yalıtım kalınlığı her zaman üst sınıra
    dayanır; maliyet amacı kalınlık için gerçek bir denge (interior optimum) üretir.
    Fizibilite: ısıtıcı, T_max'taki kaybı karşılayamıyorsa tasarım elenir (f = ∞).
    """
    hr = np.asarray(hr, dtype=float)
    t = np.asarray(t, dtype=float)
    n_mod = max(1, int(base.get("n_modules", 1)))
    r = (volume / n_mod / (np.pi * hr)) ** (1.0 / 3.0)               # her modülün yarıçapı
    h = hr * r
    ua = n_mod * ua_vec(r, h, t, base["k_ins"], base["h_out"])      # N modülün toplam UA'sı
    d_env = base["t_max"] - base["t_env"]
    feasible = base["eta_heater_pct"] / 100.0 * base["p_heat_mw"] * 1e6 > ua * d_env * 1.000001
    q_loss_kw = ua * d_env / 1000.0
    v_ins = n_mod * (np.pi * (r + t) ** 2 * (h + 2.0 * t) - np.pi * r ** 2 * h)
    t_avg = 0.5 * (base["t_max"] + base["t_min"])
    e_year_mwh = ua * (t_avg - base["t_env"]) * 8760.0 / 1e6
    cost = prices["ins"] * v_ins + prices["energy"] * e_year_mwh * prices["years"]
    f = q_loss_kw if objective == OBJ_LOSS else cost
    return np.where(feasible, f, np.inf), q_loss_kw, cost


def optimize_design(volume, base, hr_b, t_b, objective, prices, n_coarse=61):
    """
    2 değişkenli sınırlı optimizasyon (scipy gerektirmez, sağlamdır):
      1) 61×61 kaba ızgarada vektörel tarama → global en iyi bölge (yerel minimuma takılma yok)
      2) En iyi nokta etrafında 6 kez yakınlaştırma (41×41 ince ızgara, pencere ±2 adım)
    """
    xs = np.linspace(hr_b[0], hr_b[1], n_coarse)
    ts = np.linspace(t_b[0], t_b[1], n_coarse)
    F = design_objective(volume, xs[:, None], ts[None, :], base, objective, prices)[0]
    if not np.isfinite(F).any():
        return None
    i, j = np.unravel_index(np.argmin(F), F.shape)
    bx, bt = float(xs[i]), float(ts[j])
    dx, dt = float(xs[1] - xs[0]), float(ts[1] - ts[0])
    for _ in range(6):
        xl = np.linspace(max(hr_b[0], bx - 2 * dx), min(hr_b[1], bx + 2 * dx), 41)
        tl = np.linspace(max(t_b[0], bt - 2 * dt), min(t_b[1], bt + 2 * dt), 41)
        Fl = design_objective(volume, xl[:, None], tl[None, :], base, objective, prices)[0]
        if not np.isfinite(Fl).any():
            break
        i, j = np.unravel_index(np.argmin(Fl), Fl.shape)
        bx, bt = float(xl[i]), float(tl[j])
        dx, dt = float(xl[1] - xl[0]), float(tl[1] - tl[0])
    return {"hr": bx, "t": bt, "xs": xs, "ts": ts, "F": F}


def compute_optimization(base):
    """Optimizasyon sekmesindeki girdileri (session_state) okuyup sonucu tek sözlükte döndürür."""
    ss = st.session_state
    err = validate_cfg(base)
    if err:
        return {"error": f"Mevcut girdiler geçersiz: {err}"}
    c0, beta = model_params(base)
    dh = float(enthalpy(base["t_max"], c0, beta) - enthalpy(base["t_min"], c0, beta))
    volume = ss["opt_target"] * J_PER_MWH / (base["rho"] * dh)        # Q = ρ V Δh  →  V = Q/(ρ Δh)
    objective = ss["opt_obj"]
    prices = {"ins": float(ss["opt_ins_price"]), "energy": float(ss["opt_energy_price"]),
              "years": float(ss["opt_years"])}
    hr_b, t_b = tuple(ss["opt_hr"]), tuple(ss["opt_t"])
    best = optimize_design(volume, base, hr_b, t_b, objective, prices)
    if best is None:
        return {"error": "Seçilen aralıkta ısıtıcı gücü, T_max'taki ısı kaybını karşılayamıyor "
                         "(tüm tasarımlar elendi). Isıtıcı gücünü artırın veya aralığı değiştirin."}
    cfg_opt = dict(base, volume=volume, h_over_r=best["hr"], t_ins=best["t"])
    cfg_ref = dict(base, volume=volume)                                # aynı enerji, mevcut şekil
    res_opt, res_ref = compute_system(cfg_opt, "Optimum"), compute_system(cfg_ref, "Mevcut şekil")

    def view(res, cfg):
        cost = float(design_objective(volume, cfg["h_over_r"], cfg["t_ins"], base, objective, prices)[2])
        v = dict(res)
        v.update(volume=cfg["volume"], hr=cfg["h_over_r"], t_ins=cfg["t_ins"], cost_total=cost)
        return v

    tol_x, tol_t = 1e-3, 1e-3
    return {
        "error": None, "volume": volume, "hr": best["hr"], "t": best["t"], "best": best,
        "objective": objective, "prices": prices, "hr_b": hr_b, "t_b": t_b, "target": ss["opt_target"],
        "view_opt": view(res_opt, cfg_opt), "view_ref": view(res_ref, cfg_ref), "cfg_opt": cfg_opt,
        "hr_on_bound": abs(best["hr"] - hr_b[0]) < tol_x or abs(best["hr"] - hr_b[1]) < tol_x,
        "t_on_bound": abs(best["t"] - t_b[0]) < tol_t or abs(best["t"] - t_b[1]) < tol_t,
        "volume_ok": CFG_LIMITS["volume"][0] <= volume <= CFG_LIMITS["volume"][1],
    }


def build_report(systems, tcs, opt=None):
    """Aktif konfigürasyon(lar)ın tüm girdi ve çıktılarını Markdown metni olarak derler."""
    def tbl(rows):
        return "| Parametre | Değer | Birim |\n|:--|--:|:--|\n" + "\n".join(f"| {a} | {b} | {u} |" for a, b, u in rows)

    L = ["# Termal Kum Bataryası — Tasarım Raporu", "",
         f"_Oluşturulma tarihi: {datetime.now():%d.%m.%Y %H:%M}_", "",
         f"Raporlanan sistem sayısı: {len(systems)}", ""]
    for i, (s, tc) in enumerate(zip(systems, tcs), 1):
        c, cy = s["cfg"], s["cycle"]
        cp_rows = [("Özgül ısı modeli", c["cp_mode"], "–")]
        if c["cp_mode"] == CP_CONST:
            cp_rows.append(("Ortalama özgül ısı c_p", f"{c['cp']:,.0f}", "J/(kg·K)"))
        else:
            cp_rows += [("c_p,25 (25 °C'de)", f"{c['cp25']:,.0f}", "J/(kg·K)"), ("β", f"{c['beta']:.2f}", "J/(kg·K²)")]
        ci = s.get("cfg_in", c)
        if ci["charge_src"] == SRC_CSP:
            src_rows = [("Şarj kaynağı", SRC_CSP, "–"),
                        ("Toplam optik toplama alanı", f"{ci['csp_area']:,.0f}", "m²"),
                        ("DNI", f"{ci['csp_dni']:,.0f}", "W/m²"), ("Optik verim", f"{ci['csp_eta_pct']:.1f}", "%"),
                        ("Hesaplanan şarj gücü Q_solar = A·DNI·η", f"{c['p_heat_mw']:.2f}", "MW")]
        else:
            src_rows = [("Şarj kaynağı", SRC_ELEC, "–"), ("Isıtıcı gücü", f"{c['p_heat_mw']:.2f}", "MW"),
                        ("Isıtıcı verimi", f"{c['eta_heater_pct']:.1f}", "%")]
        hyb_rows = []
        if c["hybrid"]:
            hyb_rows = [("Hibrit karışım", f"%{c['mix_pct']} {c['mat1']} + %{100 - c['mix_pct']} {c['mat2']}", "–"),
                        ("Karışım oranı tabanı", c["mix_basis"], "–")]
        L += [f"## {i}. Sistem: {s['name']}", "", "### 1. Girdi parametreleri", "", tbl([
            ("Kum hacmi V", f"{c['volume']:,.1f}", "m³"), ("H/R oranı", f"{c['h_over_r']:.2f}", "–"),
            ("Malzeme", material_label(c), "–"), ("Yığın yoğunluğu ρ", f"{c['rho']:,.0f}", "kg/m³"), *cp_rows, *hyb_rows,
            ("T_max", f"{c['t_max']}", "°C"), ("T_min", f"{c['t_min']}", "°C"), ("T_çevre", f"{c['t_env']}", "°C"),
            ("Yalıtım kalınlığı t", f"{c['t_ins']:.2f}", "m"), ("Yalıtım iletkenliği k", f"{c['k_ins']:.3f}", "W/(m·K)"),
            *src_rows, ("Deşarj gücü", f"{c['p_dis_mw']:.2f}", "MW"),
            ("Silo modül sayısı N", f"{s['n_mod']}", "adet"),
            ("Dış taşınım katsayısı h_dış", f"{c['h_out']:.1f}", "W/(m²·K)"),
            ("Bekleme süresi", f"{c['hold_days']:.1f}", "gün"), ("1-D hücre sayısı N", f"{c['n_cells']}", "–"),
            ("Efektif eksenel iletkenlik k_eff", f"{c['k_eff']:.2f}", "W/(m·K)"),
            ("Şarj / deşarj kesme eşiği", f"%{c['cut_ch_pct']} / %{c['cut_dis_pct']}", "ΔT"),
        ]), "", "### 2. Geometri ve ısıl özellikler", "", tbl([
            ("Modül başına hacim V/N", f"{s['v_mod']:,.1f}", "m³"),
            ("Silo (modül) yarıçapı R", f"{s['r']:.2f}", "m"), ("Silo (modül) yüksekliği H", f"{s['h']:.2f}", "m"),
            ("Tek dev siloya göre ısı kaybı artışı", f"{s['mod_penalty_pct']:.1f}", "%"),
            ("Kum kütlesi m", f"{s['mass_t']:,.0f}", "ton"), ("Ortalama özgül ısı c̄_p", f"{s['cp_mean']:,.0f}", "J/(kg·K)"),
            ("İç yüzey alanı", f"{s['a_inner']:,.0f}", "m²"), ("Toplam ısıl iletkenlik UA", f"{s['ua']:,.1f}", "W/K"),
            ("Eşdeğer U", f"{s['u_eff']:.3f}", "W/(m²·K)"), ("Zaman sabiti τ", f"{s['tau_days']:,.0f}", "gün"),
        ]), "", "### 3. Ana sonuçlar (0-D model)", "", tbl([
            ("Depolanan ısıl enerji Q", f"{s['e_mwh']:,.1f}", "MWh"),
            ("Enerji yoğunluğu", f"{s['e_mwh'] * 1000 / c['volume']:,.0f}", "kWh/m³"),
            ("Isı kaybı (T_max'ta)", f"{s['q_loss_kw']:,.1f}", "kW"),
            ("Günlük kayıp oranı", f"{s['daily_loss_pct']:.2f}", "%/gün"),
            ("Maks. deşarj süresi (ideal)", f"{s['t_dis_h']:,.1f}", "saat"),
            ("Round-trip verim", f"{s['eta_rt_pct']:.1f}" if cy else "hesaplanamadı", "%"),
            ("Şarj süresi", f"{cy['t_charge_h']:,.1f}" if cy else "—", "saat"),
            ("Gerçek deşarj süresi", f"{cy['t_discharge_h']:,.1f}" if cy else "—", "saat"),
            ("Şarj girdi enerjisi", f"{cy['e_in_mwh']:,.1f}" if cy else "—", "MWh"),
            ("Verilen ısı", f"{cy['e_out_mwh']:,.1f}" if cy else "—", "MWh"),
        ]), "", "### 4. 1-D termoklin sonuçları", "", tbl([
            ("Kapasite kullanımı", f"{tc['util_pct']:.1f}", "%"), ("1-D round-trip verim", f"{tc['eta_pct']:.1f}", "%"),
            ("Şarj süresi", f"{tc['t_charge_h']:,.1f}", "saat"), ("Gerçek deşarj süresi", f"{tc['t_dis_h']:,.1f}", "saat"),
            ("Şarj girdi enerjisi", f"{tc['e_in_mwh']:,.1f}", "MWh"), ("Verilen ısı", f"{tc['e_out_mwh']:,.1f}", "MWh"),
            ("Termoklin kalınlığı (%50 deşarjda)", f"{tc['thick_m']:.2f}" if is_num(tc["thick_m"]) else "—", "m"),
        ]), "", "### 5. Deşarjsız bekleme (soğuma)", "",
            "| Süre (ay) | Kum sıcaklığı (°C) | Isı kaybı (kW) | Kalan faydalı enerji (%) |\n|--:|--:|--:|--:|"]
        h_hi = float(enthalpy(c["t_max"], s["c0"], s["beta"]))
        h_lo = float(enthalpy(c["t_min"], s["c0"], s["beta"]))
        for m in (1, 3, 6, 12):
            tm, lm = cooling_series(s["cool"], np.array([m * DAYS_PER_MONTH * SEC_PER_DAY]), s["ua"], c["t_env"])
            rem = max(0.0, (float(enthalpy(tm[0], s["c0"], s["beta"])) - h_lo) / (h_hi - h_lo)) * 100.0
            L.append(f"| {m} | {tm[0]:.1f} | {lm[0]:.2f} | {rem:.1f} |")
        L.append("")
    if opt and not opt.get("error"):
        vo, vr = opt["view_opt"], opt["view_ref"]
        L += ["## Optimizasyon önerisi", "",
              f"Amaç: **{opt['objective']}** · Hedef enerji: {opt['target']:,.1f} MWh · "
              f"Arama aralığı: H/R {opt['hr_b'][0]:.1f}–{opt['hr_b'][1]:.1f}, t {opt['t_b'][0]:.2f}–{opt['t_b'][1]:.2f} m", "",
              "| Büyüklük | Optimum | Mevcut şekil (aynı enerji) | Birim |\n|:--|--:|--:|:--|"]
        for key, label, fmt, unit, _, _ in OPT_SPECS:
            a, b = vo.get(key), vr.get(key)
            L.append(f"| {label} | {fmt.format(a) if is_num(a) else '—'} | {fmt.format(b) if is_num(b) else '—'} | {unit} |")
        if opt["objective"] == OBJ_COST:
            L.append(f"| Toplam maliyet | {vo['cost_total']:,.0f} | {vr['cost_total']:,.0f} | € |")
        L.append("")
    L += ["---", "## Model varsayımları (özet)", "",
          "- 0-D toplu kapasite modeli ve 1-D LTE termoklin modeli (MUSCL + SSP-RK2) birlikte kullanılmıştır.",
          "- c_p sabit veya c_p(T) = c_p,25 + β(T − 25); enerji entalpi farkından hesaplanır.",
          "- Isı kaybı: silindirik radyal iletim + düz kapaklar + dış film direnci; yalıtım k değeri sabittir. Modüler tesiste toplam UA = N·UA(V/N).",
          "- Hibrit yatakta ρ ve c_p ağırlıklı ortalamayla (ideal karışım) hesaplanır; CSP'de şarj gücü Q_solar = A·DNI·η sabit kabul edilir.",
          "- Round-trip verim elektrik → ısı → ısı verimidir (ısıdan elektriğe dönüşüm dahil değildir).",
          "- Ön tasarım aracıdır; ayrıntılı tasarım için CFD veya iki-fazlı yatak modelleri gerekir.", ""]
    return "\n".join(L)


# =============================================================================
# 3c) 3D TESİS TASARIMI: ağ (mesh) üretimi ve modüler trade-off
# =============================================================================
MAT_COLORS = {"Silika Kumu (Kuvars)": "#e9c46a", "Bazalt (kırma)": "#475569",
              "Refrakter (şamot, kırma)": "#c2410c", "Manuel Giriş": "#8b5cf6"}


def cylinder_mesh(cx, cy, r, z0, z1, n_seg=36):
    """
    Kapalı bir silindirin üçgen ağı (Plotly Mesh3d için x, y, z, i, j, k).
    Yan yüz ve kapaklar ayrı köşe kümeleri kullanır → kenarlar keskin, ışıklandırma doğru olur.
    Üçgen sıralaması dışa bakan normal verecek şekildedir.
    """
    n = n_seg
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rx, ry = cx + r * np.cos(th), cy + r * np.sin(th)
    x = np.concatenate([rx, rx, rx, rx, [cx, cx]])
    y = np.concatenate([ry, ry, ry, ry, [cy, cy]])
    z = np.concatenate([np.full(n, z0), np.full(n, z1), np.full(n, z0), np.full(n, z1), [z0, z1]])
    a = np.arange(n)
    b = (a + 1) % n
    I = np.concatenate([a, b, np.full(n, 4 * n), np.full(n, 4 * n + 1)])
    J = np.concatenate([b, n + b, 2 * n + b, 3 * n + a])
    K = np.concatenate([n + a, n + a, 2 * n + a, 3 * n + b])
    return x, y, z, I, J, K


def merge_meshes(parts):
    """Birçok silindiri tek Mesh3d izinde birleştirir (köşe indeksleri ötelenir)."""
    xs, ys, zs, Is, Js, Ks, off = [], [], [], [], [], [], 0
    for x, y, z, I, J, K in parts:
        xs.append(x); ys.append(y); zs.append(z)
        Is.append(I + off); Js.append(J + off); Ks.append(K + off)
        off += x.size
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs),
            np.concatenate(Is), np.concatenate(Js), np.concatenate(Ks))


def build_plant_figure(res, show_ins=True, n_pairs=3):
    """
    N adet modüler siloyu ızgara düzeninde çizer. Hibritte her silo, Malzeme 1 ve Malzeme 2'nin
    hacim oranına göre yüksekliği bölünmüş ALTERNATİF KATMANLAR (n_pairs çift) olarak renklendirilir.
    Silindir kesiti sabit olduğundan hacim kesri = yükseklik kesridir.
    Dönüş: (figür, silo aralığı, sütun sayısı, satır sayısı, saha genişliği, saha derinliği)
    """
    cfg, n, r, h = res["cfg"], res["n_mod"], res["r"], res["h"]
    t = cfg["t_ins"]
    pitch = 2.0 * (r + t) + max(0.35 * (r + t), 0.5)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    centers = [((i % cols) * pitch, (i // cols) * pitch) for i in range(n)]

    if cfg["hybrid"]:
        phi1 = composite_props(cfg["mat1"], cfg["mat2"], cfg["mix_pct"], cfg["mix_basis"])["phi1"]
        layers, z = [], 0.0
        for _ in range(n_pairs):
            for mname, frac in ((cfg["mat1"], phi1), (cfg["mat2"], 1.0 - phi1)):
                dz = h * frac / n_pairs
                if dz > 1e-9:
                    layers.append((mname, z, z + dz))
                    z += dz
        legend = {cfg["mat1"]: f"{cfg['mat1']} — hacimce %{phi1 * 100:.0f}",
                  cfg["mat2"]: f"{cfg['mat2']} — hacimce %{(1 - phi1) * 100:.0f}"}
    else:
        layers = [(cfg["material"], 0.0, h)]
        legend = {cfg["material"]: cfg["material"]}

    groups = {}
    for cx, cy in centers:
        for mname, z0, z1 in layers:
            groups.setdefault(mname, []).append(cylinder_mesh(cx, cy, r, z0, z1))

    fig = go.Figure()
    light = dict(ambient=0.55, diffuse=0.9, specular=0.15, roughness=0.6)
    for mname, parts in groups.items():
        x, y, z, I, J, K = merge_meshes(parts)
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, color=MAT_COLORS.get(mname, "#8b5cf6"),
                                opacity=1.0, flatshading=False, lighting=light, name=legend.get(mname, mname),
                                showlegend=True, hoverinfo="name"))
    if show_ins:
        x, y, z, I, J, K = merge_meshes([cylinder_mesh(cx, cy, r + t, -t, h + t) for cx, cy in centers])
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, color="#cbd5e1", opacity=0.16, flatshading=False,
                                name=f"Yalıtım (t = {t:.2f} m)", showlegend=True, hoverinfo="name"))
    # Zemin levhası
    m = 0.6 * pitch
    gx = [-m, (cols - 1) * pitch + m, (cols - 1) * pitch + m, -m]
    gy = [-m, -m, (rows - 1) * pitch + m, (rows - 1) * pitch + m]
    fig.add_trace(go.Mesh3d(x=gx, y=gy, z=[-t - 0.02 * h] * 4, i=[0, 0], j=[1, 2], k=[2, 3], color="#94a3b8",
                            opacity=0.35, name="Zemin", showlegend=False, hoverinfo="skip"))
    fig.update_layout(
        height=620, margin=dict(l=0, r=0, t=50, b=0), paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text=f"{n} × silo · her biri {res['v_mod']:,.0f} m³ (R = {r:.1f} m, H = {h:.1f} m)", font=dict(size=18)),
        legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="left", x=0.0),
        scene=dict(aspectmode="data", camera=dict(eye=dict(x=1.5, y=-1.7, z=0.9)),
                   xaxis=dict(title="x (m)", showbackground=False), yaxis=dict(title="y (m)", showbackground=False),
                   zaxis=dict(title="z (m)", showbackground=False)))
    width = (cols - 1) * pitch + 2.0 * (r + t)
    depth = (rows - 1) * pitch + 2.0 * (r + t)
    return fig, pitch, cols, rows, width, depth


def modular_curve(cfg, n_max=20):
    """N = 1…n_max için toplam ısı kaybı (kW, T_max'ta) ve tek siloya göre artış (%)."""
    ns = np.arange(1, n_max + 1)
    r = (cfg["volume"] / ns / (np.pi * cfg["h_over_r"])) ** (1.0 / 3.0)
    ua = ns * ua_vec(r, cfg["h_over_r"] * r, cfg["t_ins"], cfg["k_ins"], cfg["h_out"])
    q = ua * (cfg["t_max"] - cfg["t_env"]) / 1000.0
    return ns, q, (ua / ua[0] - 1.0) * 100.0


def trade_off_text(n, x_pct):
    """Modül sayısına göre dinamik mühendislik analizi metni."""
    if n == 1:
        return ("Mühendislik Analizi: Tek Dev Silo. Avantaj: Minimum yüzey alanı ve en düşük ısı kaybı. "
                "Dezavantaj: Yüksek inşaat zorluğu, artan rüzgar yükü ve termal gerilme.")
    return (f"Mühendislik Analizi: Modüler Tesis ({n} adet). Avantaj: Kolay ve ucuz inşaat, düşük rüzgar yükü, "
            f"yüksek operasyonel esneklik (bir modül bozulursa diğerleri çalışır). "
            f"Dezavantaj: Artan toplam yüzey alanı nedeniyle %{x_pct:.1f} oranında daha yüksek ısı kaybı.")


# =============================================================================
# 4) SAYFA YAPISI VE STİL
# =============================================================================
st.set_page_config(page_title="Termal Kum Bataryası Simülatörü", page_icon="🏖️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1500px;}
    .hero {padding: 1.4rem 1.8rem; border-radius: 20px; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, rgba(249,115,22,0.18), rgba(59,130,246,0.14));
        border: 1px solid rgba(148,163,184,0.30);}
    .hero h1 {margin: 0; font-size: 2rem; letter-spacing: -0.5px;}
    .hero p {margin: .3rem 0 0 0; opacity: .75;}
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(249,115,22,0.10), rgba(59,130,246,0.08));
        border: 1px solid rgba(148,163,184,0.30); border-radius: 16px; padding: 16px 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);}
    div[data-testid="stMetricValue"] {font-weight: 700;}
    .section-title {font-size: 1.05rem; font-weight: 600; opacity: .8; margin: .8rem 0 .2rem 0;}
    .sys-badge {display:inline-block; padding: .25rem .8rem; border-radius: 999px; font-weight: 600;
        border: 1px solid rgba(148,163,184,0.40); margin-bottom: .6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
        <h1>🏖️ Termal Kum Bataryası — Parametrik Tasarım ve Simülasyon</h1>
        <p>Duyulur ısı depolama · c_p(T) · Fourier ısı iletimi · Newton soğuma · 1-D termoklin · hibrit yatak · CSP · modüler tesis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 5) SESSION STATE: widget başlangıçları ve kayıtlı konfigürasyonlar
# =============================================================================
# Widget anahtarları = konfigürasyon anahtarları. Başlangıç değerleri widget'lar
# oluşturulmadan ÖNCE yazılır; böylece "yükle" geri çağrısı (callback) değerleri
# doğrudan değiştirebilir.
for _k, _v in DEFAULT_CFG.items():
    st.session_state.setdefault(_k, _v)
# Koşullu gizlenen widget'ların (ısıtıcı gücü / CSP alanları) değeri, widget çizilmediği
# çalıştırmalarda silinmesin diye burada yeniden atanır (Streamlit widget temizliğini engeller).
for _k in ("p_heat_mw", "csp_area", "csp_dni", "csp_eta_pct"):
    st.session_state[_k] = st.session_state[_k]
st.session_state.setdefault("horizon_months", 12)
st.session_state.setdefault("show_1d_cooling", True)
st.session_state.setdefault("uploader_n", 0)
# Optimizasyon sekmesi girdileri (sonuç, rapor için sayfanın başında hesaplandığından burada başlatılır)
st.session_state.setdefault("opt_target", 100.0)
st.session_state.setdefault("opt_hr", (1.0, 6.0))
st.session_state.setdefault("opt_t", (0.1, 1.0))
st.session_state.setdefault("opt_obj", OBJ_LOSS)
st.session_state.setdefault("opt_ins_price", 200.0)
st.session_state.setdefault("opt_energy_price", 60.0)
st.session_state.setdefault("opt_years", 20)
if "saved_configs" not in st.session_state:
    st.session_state["saved_configs"] = {
        "Varsayılan – Silika Kumu": dict(DEFAULT_CFG),
        "Varsayılan – Bazalt": dict(BAZALT_CFG),
    }


def sync_material_props():
    """
    Malzeme girdilerini tek yerden senkronlar:
      • Hibrit mod açıksa ρ, c_p, c_p,25, β kompozit (ağırlıklı ortalama) değerlere ayarlanır.
      • Değilse hazır malzeme seçilince preset değerleri doldurulur (Manuel'de kullanıcı değerleri korunur).
    """
    ss = st.session_state
    if ss["hybrid"]:
        p = composite_props(ss["mat1"], ss["mat2"], ss["mix_pct"], ss["mix_basis"])
    elif ss["material"] != "Manuel Giriş":
        p = MATERIALS[ss["material"]]
    else:
        return
    for k in ("rho", "cp", "cp25", "beta"):
        ss[k] = float(p[k])


on_material_change = sync_material_props


def save_config():
    """Yan menüdeki mevcut girdileri verilen isimle session_state'e kaydeder."""
    name = st.session_state.get("save_name", "").strip()
    cur = {k: st.session_state[k] for k in CFG_KEYS}
    if not name:
        st.session_state["_msg"] = ("warning", "Lütfen bir konfigürasyon adı girin.")
    elif validate_cfg(cur):
        st.session_state["_msg"] = ("error", validate_cfg(cur))
    else:
        existed = name in st.session_state["saved_configs"]
        st.session_state["saved_configs"][name] = cur
        st.session_state["_msg"] = ("success", f"'{name}' {'güncellendi' if existed else 'kaydedildi'}.")


def load_config():
    """Kayıtlı bir konfigürasyonu yan menü widget'larına geri yükler."""
    cfg = st.session_state["saved_configs"][st.session_state["load_select"]]
    for k, v in cfg.items():
        st.session_state[k] = v
    st.session_state["_msg"] = ("success", f"'{st.session_state['load_select']}' yüklendi.")


def delete_config():
    """Seçili kayıtlı konfigürasyonu listeden siler (geri alınamaz; önce JSON dışa aktarım önerilir)."""
    name = st.session_state.get("load_select")
    configs = st.session_state["saved_configs"]
    if name in configs:
        del configs[name]
        st.session_state["_msg"] = ("success", f"'{name}' silindi.")
    else:
        st.session_state["_msg"] = ("warning", "Silinecek bir kayıt seçilmedi.")


def import_configs():
    """JSON dosyasından konfigürasyonları içe aktarır; aynı isimli kayıtlar güncellenir."""
    key = f"import_file_{st.session_state['uploader_n']}"
    f = st.session_state.get(key)
    if f is None:
        return
    try:
        new, skipped = parse_import(f.getvalue())
        configs = st.session_state["saved_configs"]
        n_upd = sum(1 for n in new if n in configs)
        configs.update(new)
        extra = f", {skipped} geçersiz kayıt atlandı" if skipped else ""
        st.session_state["_msg"] = ("success", f"{len(new) - n_upd} yeni, {n_upd} güncellenen kayıt içe aktarıldı{extra}.")
    except ValueError as exc:
        st.session_state["_msg"] = ("error", str(exc))
    st.session_state["uploader_n"] += 1        # anahtarı değiştirerek yükleyiciyi sıfırlar (tekrar içe aktarmayı önler)


def apply_opt():
    """Optimizasyon önerisini (hacim, H/R, yalıtım kalınlığı) yan menü girdilerine uygular."""
    r = st.session_state.get("opt_result")
    if not r:
        return
    lo_v, hi_v = CFG_LIMITS["volume"]
    st.session_state["volume"] = int(round(min(max(r["volume"], lo_v), hi_v)))
    st.session_state["h_over_r"] = round(float(r["hr"]), 1)
    st.session_state["t_ins"] = round(round(float(r["t"]) / 0.05) * 0.05, 2)
    st.session_state["_msg"] = ("success", "Optimum tasarım yan menüye uygulandı (adım büyüklüğüne yuvarlandı).")


# Silinen/geçersiz kayıtlara işaret eden selectbox durumlarını widget'lar oluşmadan önce temizle.
_valid = list(st.session_state["saved_configs"].keys())
for _k, _opts in (("load_select", _valid), ("cmp_a", [CUR] + _valid), ("cmp_b", [CUR] + _valid)):
    if _k in st.session_state and st.session_state[_k] not in _opts:
        del st.session_state[_k]

# =============================================================================
# 6) YAN MENÜ
# =============================================================================
sb = st.sidebar
sb.title("⚙️ Tasarım Parametreleri")

# ---- Karşılaştırma modu ----------------------------------------------------
compare = sb.toggle("🔀 Karşılaştırma Modu", key="compare_mode",
                    help="İki konfigürasyonu yan yana ve ortak eksenli grafiklerle karşılaştırır.")
saved = st.session_state["saved_configs"]
if compare:
    options = [CUR] + list(saved.keys())
    st.session_state.setdefault("cmp_a", CUR)
    st.session_state.setdefault("cmp_b", "Varsayılan – Bazalt" if "Varsayılan – Bazalt" in saved
                                else (next(iter(saved), CUR)))
    sb.selectbox("Sistem A (referans)", options, key="cmp_a")
    sb.selectbox("Sistem B (Δ = B − A)", options, key="cmp_b")
    sb.caption("“Mevcut girdiler” = aşağıdaki menüde şu an ayarlı değerler. "
               "Yeni bir sistemi kaydedip karşılaştırmaya ekleyebilirsiniz.")

# ---- Silo geometrisi --------------------------------------------------------
sb.subheader("📐 Silo Geometrisi")
sb.slider("Kum hacmi V (m³)", 50, 20000, step=50, key="volume")
sb.slider("Yükseklik / Yarıçap oranı (H/R)", 0.5, 8.0, step=0.1, key="h_over_r",
          help="Kapalı silindirde yüzey alanı (dolayısıyla ısı kaybı) H/R = 2 iken minimumdur. "
               "Uzun-ince silolar termal katmanlaşma açısından avantajlıdır.")

sb.slider("Silo Modül Sayısı (N)", 1, 20, key="n_modules",
          help="Toplam hacim V, N eşit siloya bölünür. Yüzey alanı arttığı için ısı kaybı büyür; "
               "inşaat ve işletme esnekliği artar. Ayrıntı için '3D Tesis Tasarımı' sekmesine bakın.")
if st.session_state["n_modules"] > 1:
    sb.caption(f"Her modül: {st.session_state['volume'] / st.session_state['n_modules']:,.0f} m³ "
               f"({st.session_state['n_modules']} adet)")

# ---- Malzeme ---------------------------------------------------------------
sb.subheader("🪨 Depolama Malzemesi")
sb.selectbox("Malzeme", list(MATERIALS.keys()), key="material", on_change=sync_material_props,
             disabled=bool(st.session_state["hybrid"]))
sb.toggle("🧪 Hibrit Karışım Modu", key="hybrid", on_change=sync_material_props,
          help="İki malzemeyi seçilen oranda karıştırır; ρ ve c_p ağırlıklı ortalamayla hesaplanıp tüm denklemlere beslenir.")
with sb.expander("Hibrit karışım ayarları", expanded=bool(st.session_state["hybrid"])):
    st.selectbox("Malzeme 1", PRESET_NAMES, key="mat1", on_change=sync_material_props)
    st.selectbox("Malzeme 2", PRESET_NAMES, key="mat2", on_change=sync_material_props)
    st.slider("Malzeme 1 oranı (%)", 0, 100, step=5, key="mix_pct", on_change=sync_material_props,
              help="Malzeme 2 oranı = 100 − Malzeme 1 oranı.")
    st.radio("Oran tabanı", [MIX_VOL, MIX_MASS], horizontal=True, key="mix_basis", on_change=sync_material_props,
             help="Hacimce: ρ hacim ağırlıklı ortalama. Kütlece: ρ harmonik ortalama. c_p her iki durumda kütle ağırlıklı ortalamadır.")
    _mix = composite_props(st.session_state["mat1"], st.session_state["mat2"],
                           st.session_state["mix_pct"], st.session_state["mix_basis"])
    st.caption(f"Kompozit: ρ = {_mix['rho']:,.0f} kg/m³ · c̄_p = {_mix['cp']:,.0f} J/kg·K · "
               f"M1 kütle kesri %{_mix['w1'] * 100:.0f}" + ("" if st.session_state["hybrid"] else "  (mod kapalı)"))
is_manual = (st.session_state["material"] == "Manuel Giriş") and not st.session_state["hybrid"]
sb.number_input("Yığın yoğunluğu ρ (kg/m³)", 500.0, 5000.0, step=10.0, key="rho", disabled=not is_manual,
                help="Boşluklu yatağın yığın yoğunluğu (tanecik yoğunluğu × (1 − porozite)).")
sb.radio("Özgül ısı modeli", [CP_CONST, CP_VAR], key="cp_mode",
         help="c_p(T) = c_p,25 + β (T − 25 °C). Sıcaklığa bağımlı mod, enerji ve süreleri daha doğru verir.")
use_var = st.session_state["cp_mode"] == CP_VAR
sb.number_input("Ortalama özgül ısı c_p (J/kg·K)", 400.0, 2000.0, step=10.0, key="cp",
                disabled=(not is_manual) or use_var, help="Sabit modda kullanılır (T_min–T_max ortalaması).")
sb.number_input("c_p,25 — 25 °C'de özgül ısı (J/kg·K)", 400.0, 2000.0, step=10.0, key="cp25",
                disabled=(not is_manual) or (not use_var), help="c_p(T) modunda kullanılır.")
sb.number_input("β — c_p'nin sıcaklık eğimi (J/kg·K²)", 0.0, 3.0, step=0.05, key="beta", format="%.2f",
                disabled=(not is_manual) or (not use_var), help="c_p(T) modunda kullanılır.")

# ---- Sıcaklıklar -----------------------------------------------------------
sb.subheader("🌡️ Sıcaklıklar")
sb.slider("T_max — maksimum şarj sıcaklığı (°C)", 200, 800, step=10, key="t_max")
sb.slider("T_min — minimum deşarj sıcaklığı (°C)", 20, 200, step=5, key="t_min")
sb.slider("T_çevre — dış ortam sıcaklığı (°C)", -20, 45, step=1, key="t_env")

# ---- Yalıtım ---------------------------------------------------------------
sb.subheader("🧱 Yalıtım")
sb.slider("Yalıtım kalınlığı (m)", 0.05, 2.0, step=0.05, key="t_ins")
sb.number_input("Yalıtım ısıl iletkenliği k (W/m·K)", 0.015, 1.0, step=0.005, format="%.3f", key="k_ins",
                help="Yüksek sıcaklık için tipik: taş yünü 0.05–0.09, mikro gözenekli 0.025–0.04.")

# ---- Sistem güçleri --------------------------------------------------------
sb.subheader("🔌 Sistem Güçleri")
sb.radio("Şarj kaynağı", [SRC_ELEC, SRC_CSP], key="charge_src",
         help="CSP seçilirse şarj gücü Q_solar = A·DNI·η_optik formülüyle hesaplanır (sabit DNI, kesintisiz şarj).")
if st.session_state["charge_src"] == SRC_ELEC:
    sb.number_input("Isıtıcı rezistans gücü (MW)", 0.1, 100.0, step=0.5, key="p_heat_mw")
else:
    sb.number_input("Toplam optik toplama alanı (m²)", 100.0, 1000000.0, step=500.0, key="csp_area",
                    help="Fresnel lens / helyostat sahasının toplam optik açıklık alanı.")
    sb.number_input("DNI (W/m²)", 100.0, 1400.0, step=10.0, key="csp_dni",
                    help="Doğrudan normal ışınım. Tipik tasarım değeri 700–950 W/m².")
    sb.slider("Optik verim (%)", 10.0, 95.0, step=0.5, key="csp_eta_pct",
              help="Yansıtma, gölgeleme, blokaj ve alıcı kayıplarını içeren toplam optik verim.")
    sb.caption(f"☀️ Q_solar = A · DNI · η = **{csp_power_mw(st.session_state):.2f} MW**")
sb.number_input("Eşanjör deşarj gücü (MW)", 0.1, 100.0, step=0.5, key="p_dis_mw")

# ---- Gelişmiş --------------------------------------------------------------
with sb.expander("🔬 Gelişmiş ayarlar"):
    st.slider("Isıtıcı verimi (%)", 90.0, 100.0, step=0.5, key="eta_heater_pct",
              disabled=st.session_state["charge_src"] == SRC_CSP,
              help="Yalnızca elektrikli rezistans için; CSP'de optik verim kullanılır.")
    st.number_input("Dış yüzey ısı taşınım katsayısı h_dış (W/m²·K)", 3.0, 50.0, step=1.0, key="h_out",
                    help="Doğal konveksiyon + ışınım için tipik 8–15 W/m²K.")
    st.number_input("Şarj–deşarj arası bekleme süresi (gün)", 0.0, 365.0, step=1.0, key="hold_days")
    st.slider("Soğuma simülasyonu süresi (ay)", 1, 24, key="horizon_months")
    st.markdown("**1-D termoklin parametreleri**")
    st.slider("Eksenel hücre sayısı N", 20, 150, step=5, key="n_cells",
              help="Yüksek N daha ince çözünürlük, daha uzun hesap süresi demektir.")
    st.number_input("Efektif eksenel iletkenlik k_eff (W/m·K)", 0.1, 5.0, step=0.1, key="k_eff",
                    help="Kum yatağı iletimi + ısıl dağılma. Büyük değer = daha kalın termoklin.")
    st.slider("Şarj kesme eşiği (çıkış sıcaklığı, ΔT %)", 1, 30, key="cut_ch_pct",
              help="Yatak çıkışı T_min + bu yüzde·ΔT değerine ulaşınca şarj biter.")
    st.slider("Deşarj kesme eşiği (çıkış sıcaklığı, ΔT %)", 30, 95, key="cut_dis_pct",
              help="Çıkış sıcaklığı T_min + bu yüzde·ΔT değerinin altına inince deşarj biter.")
    st.checkbox("Soğuma grafiğinde 1-D (eksenel iletimli) kayıp eğrisini göster", key="show_1d_cooling")

# ---- Konfigürasyon kaydet / yükle -------------------------------------------
sb.subheader("💾 Konfigürasyonlar")
sb.text_input("Konfigürasyon adı", key="save_name", placeholder="örn. Kırma Taş – 2000 m³")
wide_call(sb.button, "Mevcut girdileri kaydet", on_click=save_config, key="btn_save")
sb.selectbox("Kayıtlı konfigürasyon", list(saved.keys()), key="load_select")
_c_load, _c_del = sb.columns(2)
with _c_load:
    wide_call(st.button, "⬆️ Yükle", on_click=load_config, key="btn_load", disabled=not saved)
with _c_del:
    wide_call(st.button, "🗑️ Sil", on_click=delete_config, key="btn_delete", disabled=not saved,
              help="Seçili kaydı kalıcı olarak siler (geri alınamaz).")
_msg = st.session_state.pop("_msg", None)
if _msg:
    getattr(sb, _msg[0])(_msg[1])
sb.caption(f"Kayıtlı konfigürasyon sayısı: {len(saved)}. Sayfa yenilenince (F5) kayıtlar silinir; "
           "kalıcı saklamak için JSON olarak indirin.")
wide_call(sb.download_button, "📤 Tümünü JSON olarak indir (Export)", data=export_json(saved),
          file_name=f"kum_bataryasi_konfigurasyonlar_{datetime.now():%Y%m%d_%H%M}.json",
          mime="application/json", key="btn_export", disabled=not saved)
sb.file_uploader("📥 JSON içe aktar (Import)", type=["json"], key=f"import_file_{st.session_state['uploader_n']}",
                 on_change=import_configs, help="Daha önce indirdiğiniz dosyayı seçin. Aynı isimli kayıtlar güncellenir.")

# =============================================================================
# 7) SİSTEMLERİN BELİRLENMESİ, DOĞRULAMA VE HESAP
# =============================================================================
cfg_cur = {k: st.session_state[k] for k in CFG_KEYS}
if compare:
    picks = [(st.session_state["cmp_a"], "A"), (st.session_state["cmp_b"], "B")]
else:
    picks = [(CUR, "")]

systems = []
for pick, _tag in picks:
    cfg = dict(DEFAULT_CFG, **(cfg_cur if pick == CUR else saved[pick]))   # eksik anahtarlara karşı güvenli
    label = f"Mevcut ({material_label(cfg)})" if pick == CUR else pick
    err = validate_cfg(cfg)
    if err:
        st.error(f"{label}: {err}")
        st.stop()
    sys_ = compute_system(effective_cfg(cfg), label)      # CSP ise şarj gücü Q_solar'dan gelir
    sys_["cfg_in"] = cfg                                  # kullanıcı girdileri (rapor için)
    systems.append(sys_)

horizon_months = st.session_state["horizon_months"]
show_1d_cool = st.session_state["show_1d_cooling"]
tcs = [tc_cycle_cached(freeze(s["cfg"])) for s in systems]

# Optimizasyon: rapor ve sekme aynı (güncel) sonucu kullansın diye burada, sekmelerden önce çözülür.
opt = compute_optimization(effective_cfg(cfg_cur))
st.session_state["opt_result"] = opt if not opt.get("error") else None

# ---- Rapor (Markdown / TXT) ----
sb.subheader("📄 Rapor Al")
_report = build_report(systems, tcs, opt).encode("utf-8")
_stamp = f"{datetime.now():%Y%m%d_%H%M}"
_r1, _r2 = sb.columns(2)
with _r1:
    wide_call(st.download_button, "Markdown (.md)", data=_report, file_name=f"kum_bataryasi_rapor_{_stamp}.md",
              mime="text/markdown", key="dl_report_md")
with _r2:
    wide_call(st.download_button, "Metin (.txt)", data=_report, file_name=f"kum_bataryasi_rapor_{_stamp}.txt",
              mime="text/plain", key="dl_report_txt")
sb.caption("Rapor, ekranda aktif olan sistem(ler)in tüm girdi ve çıktılarını içerir.")

# =============================================================================
# 8) ANA METRİK KARTLARI
# =============================================================================
if not compare:
    s0 = systems[0]
    render_cards(s0, None, MAIN_SPECS, 4)
    st.markdown('<div class="section-title">Türetilmiş tasarım büyüklükleri</div>', unsafe_allow_html=True)
    render_cards(s0, None, DERIVED_SPECS, 6)
    if s0["n_mod"] > 1:
        st.caption(f"ℹ️ Silo boyutları, yüzey alanı ve U **modül başınadır** ({s0['n_mod']} × {s0['v_mod']:,.0f} m³). "
                   f"Isı kaybı toplam yüzey nedeniyle tek dev siloya göre %{s0['mod_penalty_pct']:.1f} daha yüksektir.")
else:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f'<span class="sys-badge">🅰️ {systems[0]["name"]} — referans</span>', unsafe_allow_html=True)
        render_cards(systems[0], None, MAIN_SPECS, 2)
        render_cards(systems[0], None, DERIVED_SPECS, 3)
        if systems[0]["n_mod"] > 1:
            st.caption(f"ℹ️ {systems[0]['n_mod']} modül · boyutlar modül başına · tek silo referansına göre kayıp +%{systems[0]['mod_penalty_pct']:.1f}")
    with col_b:
        st.markdown(f'<span class="sys-badge">🅱️ {systems[1]["name"]} — Δ = B − A</span>', unsafe_allow_html=True)
        render_cards(systems[1], systems[0], MAIN_SPECS, 2)
        render_cards(systems[1], systems[0], DERIVED_SPECS, 3)
        if systems[1]["n_mod"] > 1:
            st.caption(f"ℹ️ {systems[1]['n_mod']} modül · boyutlar modül başına · tek silo referansına göre kayıp +%{systems[1]['mod_penalty_pct']:.1f}")
    st.caption("Delta renkleri mühendislik yönüne göre: enerji, süre ve verimde artış yeşil; "
               "ısı kaybı ve eşdeğer U'da artış kırmızıdır.")

for s in systems:
    if s["cycle"] is None:
        st.error(f"⚠️ {s['name']}: Şarj gücü (ısıtıcı veya CSP), T_max sıcaklığındaki ısı kaybını karşılayamıyor; "
                 "sistem hedef sıcaklığa ulaşamaz. Şarj gücünü artırın veya yalıtımı iyileştirin.")

st.write("")

# =============================================================================
# 9) SEKMELER
# =============================================================================
tab_graph, tab_tc, tab_sweep, tab_opt, tab_3d, tab_eq, tab_data = st.tabs(
    ["📊 Grafikler", "🌡️ Termoklin (1-D)", "📈 Analiz", "🎯 Optimizasyon", "🏗️ 3D Tesis Tasarımı",
     "🧮 Denklemler ve Varsayımlar", "📋 Veri & İndirme"])

# Ortak zaman ekseni (soğuma) ve sistem renkleri
horizon_s = horizon_months * DAYS_PER_MONTH * SEC_PER_DAY
t_axis = np.linspace(0, horizon_s, 500)
months_axis = t_axis / (DAYS_PER_MONTH * SEC_PER_DAY)
sys_colors = [ORANGE, BLUE]

# Her sistem için soğuma verileri
for s in systems:
    s["temp"], s["loss"] = cooling_series(s["cool"], t_axis, s["ua"], s["cfg"]["t_env"])
    if show_1d_cool:
        t1, l1 = tc_cooling_cached(freeze(s["cfg"]), horizon_months)
        s["loss_1d"] = np.interp(t_axis, t1, l1)
    else:
        s["loss_1d"] = None

# ------------------------- GRAFİKLER ----------------------------------------
with tab_graph:
    # ---- Grafik 1: Hacim vs Enerji ----
    v_max = max(2.0 * max(s["cfg"]["volume"] for s in systems), 100.0)
    v_axis = np.linspace(0, v_max, 120)

    def e_curve(rho, c0, beta, t_max, t_min):
        # E(V) = ρ [h(T_max) − h(T_min)] V / 3.6e9  → hacme göre DOĞRUSAL; eğim = hacimsel ısı kapasitesi.
        return rho * float(enthalpy(t_max, c0, beta) - enthalpy(t_min, c0, beta)) * v_axis / J_PER_MWH

    fig1 = go.Figure()
    if not compare:
        s0 = systems[0]
        cfg0 = s0["cfg"]
        for name in PRESET_NAMES:
            p = MATERIALS[name]
            c0p, bp = (p["cp"], 0.0) if cfg0["cp_mode"] == CP_CONST else (p["cp25"], p["beta"])
            sel = (name == cfg0["material"]) and not cfg0["hybrid"]
            fig1.add_trace(go.Scatter(
                x=v_axis, y=e_curve(p["rho"], c0p, bp, cfg0["t_max"], cfg0["t_min"]), mode="lines", name=name,
                line=dict(width=4 if sel else 2, color=ORANGE if sel else GRAY, dash="solid" if sel else "dot"),
                opacity=1.0 if sel else 0.7,
                hovertemplate="V = %{x:,.0f} m³<br>E = %{y:,.1f} MWh<extra>" + name + "</extra>"))
        if cfg0["hybrid"] or cfg0["material"] == "Manuel Giriş":
            fig1.add_trace(go.Scatter(
                x=v_axis, y=e_curve(cfg0["rho"], s0["c0"], s0["beta"], cfg0["t_max"], cfg0["t_min"]),
                mode="lines", name=material_label(cfg0), line=dict(width=4, color=ORANGE)))
        marks = [(s0, BLUE, "Mevcut tasarım")]
    else:
        for s, colr in zip(systems, sys_colors):
            c = s["cfg"]
            fig1.add_trace(go.Scatter(
                x=v_axis, y=e_curve(c["rho"], s["c0"], s["beta"], c["t_max"], c["t_min"]), mode="lines",
                name=s["name"], line=dict(width=4, color=colr),
                hovertemplate="V = %{x:,.0f} m³<br>E = %{y:,.1f} MWh<extra>" + s["name"] + "</extra>"))
        marks = [(s, colr, s["name"]) for s, colr in zip(systems, sys_colors)]
    for s, colr, lbl in marks:
        fig1.add_trace(go.Scatter(
            x=[s["cfg"]["volume"]], y=[s["e_mwh"]], mode="markers+text", name=f"Tasarım noktası: {lbl}",
            marker=dict(size=14, color=colr, line=dict(width=2, color="white")),
            text=[f"{s['e_mwh']:,.0f} MWh"], textposition="top left", showlegend=compare,
            hovertemplate="V = %{x:,.0f} m³<br>E = %{y:,.1f} MWh<extra></extra>"))
    fig1.update_xaxes(title="Kum hacmi V (m³)")
    fig1.update_yaxes(title="Depolanan ısıl enerji (MWh)")
    style_fig(fig1, "Grafik 1 — Hacim vs. Depolanan Enerji")
    wide_call(st.plotly_chart, fig1, key="fig_energy")

    # ---- Grafik 2: Zaman vs Isı Kaybı (Newton soğuma, c_p(T)'li) ----
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    for idx, (s, colr) in enumerate(zip(systems, sys_colors)):
        c = s["cfg"]
        pre = f"{s['name']} · " if compare else ""
        loss_color, temp_color = (ORANGE, BLUE) if not compare else (colr, colr)
        fig2.add_trace(go.Scatter(
            x=months_axis, y=s["loss"], name=f"{pre}Isı kaybı (kW)", mode="lines",
            line=dict(color=loss_color, width=4),
            fill="tozeroy" if not compare else None, fillcolor="rgba(249,115,22,0.12)",
            hovertemplate="%{y:,.2f} kW<extra>" + pre + "Isı kaybı</extra>"), secondary_y=False)
        if s["loss_1d"] is not None:
            fig2.add_trace(go.Scatter(
                x=months_axis, y=s["loss_1d"], name=f"{pre}1-D kayıp (kW)", mode="lines",
                line=dict(color=loss_color, width=2, dash="dashdot"),
                hovertemplate="%{y:,.2f} kW<extra>" + pre + "1-D kayıp</extra>"), secondary_y=False)
        fig2.add_trace(go.Scatter(
            x=months_axis, y=s["temp"], name=f"{pre}Kum sıcaklığı (°C)", mode="lines",
            line=dict(color=temp_color, width=3, dash="dash"),
            hovertemplate="%{y:,.0f} °C<extra>" + pre + "Kum sıcaklığı</extra>"), secondary_y=True)
        fig2.add_trace(go.Scatter(
            x=[0, months_axis[-1]], y=[c["t_min"]] * 2, name=f"{pre}T_min", mode="lines",
            line=dict(color=GREEN if not compare else colr, width=2, dash="dot"), hoverinfo="skip"),
            secondary_y=True)
        # T = T_min anı: soğuma tablosundan ters interpolasyon (T azalan → diziler ters çevrilir)
        if c["t_min"] - c["t_env"] > 0.05:
            t_reach = float(np.interp(c["t_min"], s["cool"]["T"][::-1], s["cool"]["t"][::-1]))
            t_reach_m = t_reach / (DAYS_PER_MONTH * SEC_PER_DAY)
            if t_reach_m <= horizon_months:
                fig2.add_vline(x=t_reach_m, line_dash="dot", line_color=GREEN if not compare else colr,
                               annotation_text=f"T = T_min ({t_reach_m:.1f} ay)",
                               annotation_position="top left" if idx == 0 else "top right")
    fig2.update_xaxes(title="Zaman (ay)")
    fig2.update_yaxes(title_text="Isı kaybı (kW)", secondary_y=False)
    fig2.update_yaxes(title_text="Kum sıcaklığı (°C)", secondary_y=True, showgrid=False)
    style_fig(fig2, "Grafik 2 — Zaman vs. Isı Kaybı (deşarj yok, Newton soğuma eğrisi)", height=520)
    wide_call(st.plotly_chart, fig2, key="fig_cooling")

    # ---- Özet tablo ----
    st.markdown('<div class="section-title">Bekleme süresine göre depo durumu</div>', unsafe_allow_html=True)
    rows = []
    for s in systems:
        c = s["cfg"]
        h_hi = float(enthalpy(c["t_max"], s["c0"], s["beta"]))
        h_lo = float(enthalpy(c["t_min"], s["c0"], s["beta"]))
        for m in (1, 3, 6, 12):
            temp_m, loss_m = cooling_series(s["cool"], np.array([m * DAYS_PER_MONTH * SEC_PER_DAY]), s["ua"], c["t_env"])
            # Kalan faydalı enerji oranı: [h(T) − h(T_min)] / [h(T_max) − h(T_min)]
            rem = max(0.0, (float(enthalpy(temp_m[0], s["c0"], s["beta"])) - h_lo) / (h_hi - h_lo)) * 100.0
            row = {"Süre (ay)": m, "Kum sıcaklığı (°C)": round(float(temp_m[0]), 1),
                   "Isı kaybı (kW)": round(float(loss_m[0]), 2), "Kalan faydalı enerji (%)": round(rem, 1)}
            rows.append({"Sistem": s["name"], **row} if compare else row)
    df_tbl = pd.DataFrame(rows).set_index(["Sistem", "Süre (ay)"] if compare else "Süre (ay)")
    wide_call(st.dataframe, df_tbl)

# ------------------------- TERMOKLİN (1-D) ----------------------------------
def render_tc(res, tc, ref_tc, ncols, key):
    """Bir sistemin 1-D termoklin kartlarını ve iki grafiğini çizer."""
    view = dict(tc)
    if ref_tc is None:
        # Tek sistem: 1-D verimi 0-D ideal değerle kıyasla (yüzde puan farkı)
        view["sub"] = {"eta_pct": (f"{tc['eta_pct'] - res['eta_rt_pct']:+.1f} p.p. (0-D'ye göre)"
                                   if is_num(res["eta_rt_pct"]) else None)}
    render_cards(view, ref_tc, TC_SPECS, ncols)

    figp = go.Figure()
    for i, (lbl, prof) in enumerate(tc["profiles"]):
        figp.add_trace(go.Scatter(x=tc["x_h"], y=prof, mode="lines", name=lbl,
                                  line=dict(width=3, color=PROFILE_COLORS[i % len(PROFILE_COLORS)],
                                            dash="dash" if lbl == "Bekleme sonu" else "solid")))
    figp.update_xaxes(title="Silo tabanından yükseklik (m)")
    figp.update_yaxes(title="Kum sıcaklığı (°C)")
    style_fig(figp, "Eksenel sıcaklık profilleri", height=420)
    figp.update_layout(hovermode="closest")
    wide_call(st.plotly_chart, figp, key=f"fig_prof_{key}")

    if tc["dis_t_h"].size:
        figd = make_subplots(specs=[[{"secondary_y": True}]])
        figd.add_trace(go.Scatter(x=tc["dis_t_h"], y=tc["dis_Tout"], name="Çıkış sıcaklığı (°C)",
                                  line=dict(color=ORANGE, width=3)), secondary_y=False)
        figd.add_trace(go.Scatter(x=tc["dis_t_h"], y=tc["dis_P_kw"], name="Verilen ısıl güç (kW)",
                                  line=dict(color=BLUE, width=3, dash="dash")), secondary_y=True)
        figd.update_xaxes(title="Deşarj süresi (saat)")
        figd.update_yaxes(title_text="Çıkış sıcaklığı (°C)", secondary_y=False)
        figd.update_yaxes(title_text="Güç (kW)", secondary_y=True, showgrid=False)
        style_fig(figd, "Deşarj: çıkış sıcaklığı ve güç", height=380)
        wide_call(st.plotly_chart, figd, key=f"fig_dis_{key}")
    else:
        st.info("Bu girdilerle deşarj fazı oluşmadı (çıkış sıcaklığı eşiğin altında).")


with tab_tc:
    st.markdown(
        "Yatak eksen boyunca hücrelere bölünür; sıcak akışkan üstten girerek **termoklin** cephesini "
        "aşağı doğru iter, deşarjda soğuk akışkan alttan girer. 0-D modelin 'ideal karışık' "
        "varsayımına göre **kapasite kullanımı ve verim** daha gerçekçi hesaplanır.")
    if not compare:
        render_tc(systems[0], tcs[0], None, 5, "single")
    else:
        ca, cb = st.columns(2)
        with ca:
            st.markdown(f'<span class="sys-badge">🅰️ {systems[0]["name"]}</span>', unsafe_allow_html=True)
            render_tc(systems[0], tcs[0], None, 2, "a")
        with cb:
            st.markdown(f'<span class="sys-badge">🅱️ {systems[1]["name"]} — Δ = B − A</span>', unsafe_allow_html=True)
            render_tc(systems[1], tcs[1], tcs[0], 2, "b")

# ------------------------- ANALİZ (PARAMETRE TARAMASI) ------------------------
with tab_sweep:
    st.markdown(
        "Seçilen parametreyi bir aralıkta tarar ve **round-trip verim** ile **ısı kaybı** değişimini "
        "aynı grafikte gösterir. Hacim dahil diğer tüm girdiler sabit tutulur (H/R taramasında hacim sabit, "
        "yalnızca şekil değişir).")
    sw1, sw2, sw3 = st.columns([1.1, 2, 1])
    sw_var = sw1.radio("Taranacak parametre", ["H/R oranı", "Yalıtım kalınlığı (m)"], key="sw_var")
    if sw_var == "H/R oranı":
        sw_lo, sw_hi = sw2.slider("H/R aralığı (min – max)", 0.5, 8.0, (1.0, 6.0), step=0.1, key="sw_hr")
        sw_key, sw_label = "h_over_r", "H/R oranı"
    else:
        sw_lo, sw_hi = sw2.slider("Yalıtım kalınlığı aralığı (m)", 0.05, 2.0, (0.1, 1.5), step=0.05, key="sw_t")
        sw_key, sw_label = "t_ins", "Yalıtım kalınlığı (m)"
    sw_n = sw3.slider("Nokta sayısı", 5, 60, 25, key="sw_n")
    use_1d_sw = st.checkbox("1-D termoklin verimini de hesapla (daha yavaş; kaba ızgarada)", key="sw_1d",
                            help="Katmanlaşma etkisi (uzun-ince silolar) yalnızca 1-D modelde görünür.")
    sw_n1 = st.slider("1-D için nokta sayısı", 3, 15, 8, key="sw_n1") if use_1d_sw else 8

    xs_sw = np.linspace(sw_lo, sw_hi, sw_n)
    xs_sw1 = np.linspace(sw_lo, sw_hi, sw_n1)
    sweep = []
    prog = st.progress(0.0, text="Tarama hesaplanıyor…") if use_1d_sw else None
    total_steps = len(systems) * sw_n1
    done = 0
    for s in systems:
        eta0, loss0, daily0 = [], [], []
        for x in xs_sw:
            r_ = compute_system(dict(s["cfg"], **{sw_key: float(x)}), s["name"])
            eta0.append(r_["eta_rt_pct"]); loss0.append(r_["q_loss_kw"]); daily0.append(r_["daily_loss_pct"])
        eta1 = None
        if use_1d_sw:
            eta1 = []
            for x in xs_sw1:
                eta1.append(tc_cycle_cached(freeze(dict(s["cfg"], **{sw_key: float(x)})))["eta_pct"])
                done += 1
                prog.progress(done / total_steps, text="1-D termoklin taraması…")
        sweep.append({"name": s["name"], "eta": np.array(eta0), "loss": np.array(loss0),
                      "daily": np.array(daily0), "eta1d": eta1, "cur": s["cfg"][sw_key]})
    if prog is not None:
        prog.empty()

    fig_sw = make_subplots(specs=[[{"secondary_y": True}]])
    for idx, sw in enumerate(sweep):
        colr = sys_colors[idx]
        pre = f"{sw['name']} · " if compare else ""
        eta_col, loss_col = (ORANGE, BLUE) if not compare else (colr, colr)
        fig_sw.add_trace(go.Scatter(x=xs_sw, y=sw["eta"], mode="lines", name=f"{pre}Round-trip verim, 0-D (%)",
                                    line=dict(color=eta_col, width=4),
                                    hovertemplate="%{y:.2f} %<extra>" + pre + "verim 0-D</extra>"), secondary_y=False)
        if sw["eta1d"] is not None:
            fig_sw.add_trace(go.Scatter(x=xs_sw1, y=sw["eta1d"], mode="lines+markers", name=f"{pre}Round-trip verim, 1-D (%)",
                                        line=dict(color=eta_col, width=2, dash="dot"), marker=dict(size=8),
                                        hovertemplate="%{y:.2f} %<extra>" + pre + "verim 1-D</extra>"), secondary_y=False)
        fig_sw.add_trace(go.Scatter(x=xs_sw, y=sw["loss"], mode="lines", name=f"{pre}Isı kaybı (kW)",
                                    line=dict(color=loss_col, width=3, dash="dash"),
                                    hovertemplate="%{y:.2f} kW<extra>" + pre + "ısı kaybı</extra>"), secondary_y=True)
        k_min = int(np.nanargmin(sw["loss"]))
        fig_sw.add_trace(go.Scatter(x=[xs_sw[k_min]], y=[sw["loss"][k_min]], mode="markers", showlegend=False,
                                    marker=dict(size=13, symbol="star", color=loss_col, line=dict(width=1, color="white")),
                                    hovertemplate=f"Min. kayıp: {sw['loss'][k_min]:.2f} kW @ {xs_sw[k_min]:.2f}<extra></extra>"),
                         secondary_y=True)
        if sw_lo <= sw["cur"] <= sw_hi:
            fig_sw.add_vline(x=sw["cur"], line_dash="dot", line_color=eta_col,
                             annotation_text=f"{pre}mevcut = {sw['cur']:.2f}",
                             annotation_position="top left" if idx == 0 else "top right")
    fig_sw.update_xaxes(title=sw_label)
    fig_sw.update_yaxes(title_text="Round-trip verim (%)", secondary_y=False)
    fig_sw.update_yaxes(title_text="Isı kaybı (kW, T_max'ta)", secondary_y=True, showgrid=False)
    style_fig(fig_sw, f"Duyarlılık analizi — {sw_label} vs. verim ve ısı kaybı", height=520)
    wide_call(st.plotly_chart, fig_sw, key="fig_sweep")
    st.caption("★ = taranan aralıkta en düşük ısı kaybı. Sabit hacimde yüzey alanı H/R ≈ 2'de minimumdur; "
               "ancak uzun-ince silolar 1-D modelde termoklin etkisini azaltabilir.")

    rows_sw = []
    for sw in sweep:
        for k_, x in enumerate(xs_sw):
            row = {sw_label: round(float(x), 3), "Round-trip verim 0-D (%)": round(float(sw["eta"][k_]), 2),
                   "Isı kaybı (kW)": round(float(sw["loss"][k_]), 2), "Günlük kayıp (%/gün)": round(float(sw["daily"][k_]), 3)}
            rows_sw.append({"Sistem": sw["name"], **row} if compare else row)
    df_sw = pd.DataFrame(rows_sw)
    with st.expander("Tarama verisi (tablo ve CSV)"):
        wide_call(st.dataframe, df_sw)
        st.download_button("⬇️ Tarama verisini CSV indir", data=df_sw.to_csv(index=False).encode("utf-8-sig"),
                           file_name="kum_bataryasi_tarama.csv", mime="text/csv", key="dl_sweep")

# ------------------------- OPTİMİZASYON ----------------------------------------
with tab_opt:
    st.markdown(
        "Hedef enerjiyi girin; araç, **mevcut yan menü girdilerini** (malzeme, sıcaklıklar, güçler, yalıtım k değeri) "
        "sabit tutarak gerekli hacmi hesaplar ve seçilen amaç için **optimum H/R ve yalıtım kalınlığını** arar.")
    o1, o2, o3 = st.columns(3)
    o1.number_input("Hedef depolanan enerji (MWh)", 1.0, 5000.0, step=10.0, key="opt_target")
    o2.slider("H/R arama aralığı", 0.5, 8.0, step=0.1, key="opt_hr")
    o3.slider("Yalıtım kalınlığı arama aralığı (m)", 0.05, 2.0, step=0.05, key="opt_t")
    st.radio("Amaç fonksiyonu", [OBJ_LOSS, OBJ_COST], key="opt_obj", horizontal=True,
             help="Yalnızca ısı kaybı minimize edilirse yalıtım kalınlığı doğal olarak üst sınıra dayanır. "
                  "Maliyet amacı, yalıtım bedeli ile ömür boyu kayıp ısının değeri arasında denge kurar.")
    cost_on = st.session_state["opt_obj"] == OBJ_COST
    p1, p2, p3 = st.columns(3)
    p1.number_input("Yalıtım birim maliyeti (€/m³)", 0.0, 100000.0, step=10.0, key="opt_ins_price", disabled=not cost_on)
    p2.number_input("Kayıp ısının değeri (€/MWh)", 0.0, 5000.0, step=5.0, key="opt_energy_price", disabled=not cost_on)
    p3.number_input("Ekonomik ömür (yıl)", 1, 60, step=1, key="opt_years", disabled=not cost_on)

    if opt.get("error"):
        st.warning(opt["error"])
    else:
        st.markdown(f'<div class="section-title">Sonuç: {opt["target"]:,.0f} MWh için optimum tasarım</div>', unsafe_allow_html=True)
        specs = OPT_SPECS + ([OPT_COST_SPEC] if cost_on else [])
        render_cards(opt["view_opt"], opt["view_ref"], specs, 4)
        st.caption("Delta (Δ) = optimum − mevcut şekil. Mevcut şekil: aynı hacim ve enerji, yan menüdeki H/R ve yalıtım kalınlığı.")
        if opt["t_on_bound"]:
            st.info("ℹ️ Yalıtım kalınlığı arama aralığının **sınırına dayandı**. Bu, seçilen amaç için daha kalın/ince "
                    "yalıtımın da iyi olabileceği anlamına gelir; aralığı genişletin veya maliyet amacını deneyin.")
        if opt["hr_on_bound"]:
            st.info("ℹ️ H/R arama aralığının sınırına dayandı; aralığı genişletmeyi deneyin.")
        if not opt["volume_ok"]:
            st.warning(f"Gerekli hacim ({opt['volume']:,.0f} m³) yan menü sınırlarının dışında; "
                       "optimum yan menüye uygulanamaz.")

        best = opt["best"]
        Z = np.where(np.isfinite(best["F"]), best["F"], np.nan)
        unit_z = "€" if cost_on else "kW"
        fig_o = go.Figure(go.Contour(x=best["xs"], y=best["ts"], z=Z.T, colorscale="Viridis", reversescale=True,
                                     colorbar=dict(title=unit_z), contours=dict(coloring="heatmap", showlines=True),
                                     hovertemplate="H/R = %{x:.2f}<br>t = %{y:.2f} m<br>f = %{z:,.1f} " + unit_z + "<extra></extra>"))
        fig_o.add_trace(go.Scatter(x=[opt["hr"]], y=[opt["t"]], mode="markers", name="Optimum",
                                   marker=dict(size=16, symbol="star", color="#ef4444", line=dict(width=2, color="white"))))
        fig_o.add_trace(go.Scatter(x=[cfg_cur["h_over_r"]], y=[cfg_cur["t_ins"]], mode="markers", name="Mevcut girdiler",
                                   marker=dict(size=12, symbol="x", color="white", line=dict(width=2, color="black"))))
        fig_o.update_xaxes(title="H/R oranı", range=[opt["hr_b"][0], opt["hr_b"][1]])
        fig_o.update_yaxes(title="Yalıtım kalınlığı (m)", range=[opt["t_b"][0], opt["t_b"][1]])
        style_fig(fig_o, f"Amaç fonksiyonu haritası ({'toplam maliyet' if cost_on else 'ısı kaybı'}, {unit_z})", height=500)
        fig_o.update_layout(hovermode="closest")
        wide_call(st.plotly_chart, fig_o, key="fig_opt")
        wide_call(st.button, "✅ Optimumu yan menü girdilerine uygula", on_click=apply_opt, key="btn_apply_opt",
                  disabled=not opt["volume_ok"])

# ------------------------- 3D TESİS TASARIMI -----------------------------------
with tab_3d:
    st.markdown(
        "Tek dev silo mu, yoksa çok modüllü bir **batarya tarlası** mı? Yan menüdeki **Silo Modül Sayısı (N)** "
        "toplam hacmi N eşit siloya böler (aynı H/R ve yalıtım). Enerji değişmez; toplam dış yüzey alanı "
        "arttığı için ısı kaybı büyür.")
    idx3 = 0
    if compare:
        idx3 = 0 if st.radio("Gösterilen sistem", ["A", "B"], horizontal=True, key="v3d_sys") == "A" else 1
    s3 = systems[idx3]
    c3, n3 = s3["cfg"], s3["n_mod"]
    show_ins = st.checkbox("Yalıtım katmanını göster", value=True, key="v3d_ins")
    fig3d, pitch3, cols3, rows3, width3, depth3 = build_plant_figure(s3, show_ins)
    wide_call(st.plotly_chart, fig3d, key="fig_3d")
    if c3["hybrid"]:
        st.caption("Görselleştirme: hibrit karışımın alternatif katman gösterimi (3 çift; kalınlıklar hacim oranına göre). "
                   "Hesaplar homojen karışım varsayar.")

    # ---- Dinamik mühendislik analizi (trade-off) ----
    msg = trade_off_text(n3, s3["mod_penalty_pct"])
    (st.success if n3 == 1 else st.info)(msg)

    # ---- Sayısal özet ----
    ns_, q_curve, inc_curve = modular_curve(c3)
    q_single = float(q_curve[0])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Modül sayısı × hacim", f"{n3} × {s3['v_mod']:,.0f} m³")
    k2.metric("Toplam iç yüzey alanı", f"{s3['a_inner']:,.0f} m²",
              delta=(f"{s3['a_inner'] - s3['a_single']:+,.0f} m² (tek siloya göre)" if n3 > 1 else None),
              delta_color="inverse", help="A_toplam = N · (2πRH + 2πR²), R ve H modül boyutlarıdır.")
    k3.metric("Isı kaybı (T_max'ta)", f"{s3['q_loss_kw']:,.1f} kW",
              delta=(f"{s3['q_loss_kw'] - q_single:+,.1f} kW · %{s3['mod_penalty_pct']:.1f}" if n3 > 1 else None),
              delta_color="inverse", help="Q_kayıp = N · UA(V/N) · (T_max − T_çevre)")
    k4.metric("Saha ayak izi (yaklaşık)", f"{width3 * depth3:,.0f} m²",
              help=f"{cols3} × {rows3} ızgara, silo merkez aralığı {pitch3:.1f} m.")

    # ---- Trade-off eğrisi: ısı kaybı vs modül sayısı ----
    fig_n = make_subplots(specs=[[{"secondary_y": True}]])
    fig_n.add_trace(go.Bar(x=ns_, y=q_curve, name="Isı kaybı (kW)",
                           marker_color=[ORANGE if int(n_) == n3 else "rgba(148,163,184,0.55)" for n_ in ns_],
                           hovertemplate="N = %{x}<br>%{y:,.1f} kW<extra></extra>"), secondary_y=False)
    fig_n.add_trace(go.Scatter(x=ns_, y=inc_curve, name="Tek siloya göre artış (%)", mode="lines+markers",
                               line=dict(color=BLUE, width=3), hovertemplate="N = %{x}<br>+%{y:.1f} %<extra></extra>"),
                    secondary_y=True)
    fig_n.update_xaxes(title="Silo modül sayısı N", dtick=1)
    fig_n.update_yaxes(title_text="Isı kaybı (kW, T_max'ta)", secondary_y=False)
    fig_n.update_yaxes(title_text="Artış (%)", secondary_y=True, showgrid=False)
    style_fig(fig_n, f"Trade-off: modül sayısı vs. ısı kaybı (V = {c3['volume']:,.0f} m³ sabit)", height=400)
    wide_call(st.plotly_chart, fig_n, key="fig_modular")

# ------------------------- DENKLEMLER ---------------------------------------
with tab_eq:
    st.markdown("### Kullanılan denklemler")
    st.markdown("**Sıcaklığa bağımlı özgül ısı ve duyulur ısı depolama**")
    st.latex(r"c_p(T)=c_{p,25}+\beta\,(T-25),\qquad h(T)=\int c_p\,dT=c_{p,25}(T-25)+\tfrac{\beta}{2}(T-25)^2")
    st.latex(r"Q=m\,[\,h(T_{max})-h(T_{min})\,]=\rho\,V\,\bar{c}_p\,(T_{max}-T_{min}),\qquad \bar{c}_p=\frac{h(T_{max})-h(T_{min})}{T_{max}-T_{min}}")
    st.markdown("**Silindirik silo geometrisi**")
    st.latex(r"V=\pi R^2 H,\qquad H=\left(\tfrac{H}{R}\right)R \;\Rightarrow\; R=\left[\frac{V}{\pi\,(H/R)}\right]^{1/3}")
    st.markdown("**Yan duvar (radyal) ve kapak (düz) ısıl dirençleri — Fourier iletimi**")
    st.latex(r"R_{yan}=\frac{\ln(r_o/r_i)}{2\pi k L}+\frac{1}{h_{d}A_{d}},\qquad R_{kapak}=\frac{t}{kA}+\frac{1}{h_{d}A_{d}}")
    st.latex(r"UA = \frac{1}{R_{yan}}+\frac{2}{R_{kapak}},\qquad \dot{Q}_{kayip}=UA\,(T_{kum}-T_{cevre})")
    st.markdown("**Deşarjsız soğuma (Newton) — c_p(T) ile genelleştirilmiş**")
    st.latex(r"m\,c_p(T)\,\frac{dT}{dt}=-UA\,(T-T_{\infty})\;\Rightarrow\; t(T)=\frac{m}{UA}\int_{T}^{T_{max}}\frac{c_p(T')}{T'-T_\infty}\,dT'")
    st.latex(r"c_p=\text{sabit}:\quad T(t)=T_\infty+(T_{max}-T_\infty)\,e^{-t/\tau},\qquad \tau=\frac{m\,c_p}{UA}")
    st.markdown("**Çevrim süreleri ve round-trip verim (0-D)**")
    st.latex(r"t_{sarj}=m\!\int_{T_{min}}^{T_{max}}\!\frac{c_p(T)\,dT}{\eta_h P_h-UA\,(T-T_\infty)},\qquad t_{desarj}=m\!\int_{T_{min}}^{T_b}\!\frac{c_p(T)\,dT}{P_d+UA\,(T-T_\infty)}")
    st.latex(r"t_{desarj}^{ideal}=\frac{Q}{P_d},\qquad \eta_{RT}=\frac{P_d\,t_{desarj}}{P_h\,t_{sarj}}")
    st.markdown("**1-D termoklin (LTE adveksiyon–difüzyon)**")
    st.latex(r"\rho\,c_p(T)\,\frac{\partial T}{\partial t}+\frac{\dot C}{A}\frac{\partial T}{\partial z}=\frac{\partial}{\partial z}\!\left(k_{eff}\frac{\partial T}{\partial z}\right)-\frac{UA_{yan}}{A\,H}\,(T-T_\infty)")
    st.latex(r"\dot C_{sarj}=\frac{\eta_h P_h}{T_{max}-T_{min}},\quad \dot C_{desarj}=\frac{P_d}{T_{max}-T_{min}},\quad P_{verilen}(t)=\dot C\,\big(T_{cikis}-T_{min}\big)")

    st.markdown("**Hibrit (kompozit) yatak — ağırlıklı ortalama**")
    st.latex(r"\rho_{k}=\varphi_1\rho_1+\varphi_2\rho_2,\qquad w_1=\frac{\varphi_1\rho_1}{\rho_{k}},\qquad c_{p,k}(T)=w_1\,c_{p,1}(T)+w_2\,c_{p,2}(T)")
    st.markdown("**CSP şarj gücü**")
    st.latex(r"Q_{solar}=A_{opt}\cdot DNI\cdot\eta_{opt}\;\;\Rightarrow\;\; P_{h}\leftarrow Q_{solar},\;\;\eta_h=1")
    st.markdown("**Modüler tesis — ısı kaybı**")
    st.latex(r"\dot Q_{kayip}=\sum_{i=1}^{N}U_iA_i\,(T-T_\infty)=N\cdot UA\!\left(\tfrac{V}{N}\right)(T-T_\infty),\qquad \Delta\%=\left(\frac{N\,UA(V/N)}{UA(V)}-1\right)\times100")
    st.markdown("**Optimizasyon (sabit hacim, iki değişken: H/R ve t)**")
    st.latex(r"V^{*}=\frac{Q_{hedef}}{\rho\,[\,h(T_{max})-h(T_{min})\,]},\qquad \min_{H/R,\;t}\; f=UA\,(T_{max}-T_\infty)")
    st.latex(r"f_{maliyet}=c_{ins}\,V_{ins}+c_E\,E_{kayip,yil}\,N,\quad V_{ins}=\pi (R+t)^2 (H+2t)-\pi R^2 H,\quad E_{kayip,yil}=UA\,(\bar T-T_\infty)\,8760")

    st.markdown("### 📖 Değişken Sözlüğü")
    st.caption("Yukarıdaki denklemlerde ve uygulamada kullanılan tüm semboller.")
    st.markdown("**Enerji ve malzeme**")
    st.markdown(md_table([
        ("Q", "Depolanan toplam termal enerji (kullanılabilir, T_min üstü)", "J veya MWh"),
        ("m", "Kum kütlesi, m = ρ·V", "kg"),
        ("ρ", "Kum yatağının yığın yoğunluğu (boşluklar dahil)", "kg/m³"),
        ("V", "Silodaki kum hacmi", "m³"),
        ("c_p", "Özgül ısı kapasitesi (sabit modda ortalama değer)", "J/(kg·K)"),
        ("c_p,25", "25 °C'deki özgül ısı (c_p(T) modeli başlangıç değeri)", "J/(kg·K)"),
        ("β", "c_p'nin sıcaklıkla lineer artış eğimi", "J/(kg·K²)"),
        ("c̄_p", "T_min–T_max aralığındaki ortalama özgül ısı", "J/(kg·K)"),
        ("h(T)", "Özgül duyulur entalpi (c_p'nin sıcaklık integrali)", "J/kg"),
        ("ΔT", "Kullanılabilir sıcaklık aralığı, T_max − T_min", "K"),
        ("T_max", "Maksimum şarj sıcaklığı", "°C"),
        ("T_min", "Minimum deşarj sıcaklığı (altında faydalı güç alınamaz)", "°C"),
        ("T_∞ (T_çevre)", "Dış ortam sıcaklığı", "°C"),
        ("T_kum", "Kum sıcaklığı (0-D modelde tek değer)", "°C"),
    ]))
    st.markdown("**Geometri**")
    st.markdown(md_table([
        ("R", "Silo (kum) yarıçapı", "m"),
        ("H", "Silo (kum) yüksekliği", "m"),
        ("H/R", "Yükseklik/yarıçap oranı (yüzey alanı minimumu H/R = 2)", "–"),
        ("r_i, r_o", "Yalıtımın iç ve dış yarıçapı (r_o = r_i + t)", "m"),
        ("L", "Silindirik yan yüzey uzunluğu (= H)", "m"),
        ("A", "Isı akışına dik yüzey alanı (kapak için πR²)", "m²"),
        ("A_iç", "Toplam iç yüzey alanı, 2πRH + 2πR²", "m²"),
        ("A_d", "Dış yüzey alanı (yalıtım dışı)", "m²"),
    ]))
    st.markdown("**Isı transferi**")
    st.markdown(md_table([
        ("t", "Yalıtım kalınlığı", "m"),
        ("k", "Yalıtımın ısıl iletkenliği", "W/(m·K)"),
        ("h_d", "Dış yüzey ısı taşınım katsayısı (konveksiyon + ışınım)", "W/(m²·K)"),
        ("R_yan, R_kapak", "Yan duvar ve kapak toplam ısıl direnci", "K/W"),
        ("UA", "Toplam ısıl iletkenlik (tüm yüzeyler paralel)", "W/K"),
        ("U", "Eşdeğer ısı geçirgenlik katsayısı, U = UA / A_iç", "W/(m²·K)"),
        ("Q̇_kayıp", "Çevreye olan ısı kaybı gücü, UA·(T_kum − T_çevre)", "W veya kW"),
        ("C", "Toplam ısıl kapasite, m·c_p", "J/K"),
        ("τ", "Zaman sabiti, C/UA; sıcaklık farkının 1/e'ye düşme süresi", "s veya gün"),
        ("θ", "Çevreye göre sıcaklık farkı, T − T_∞", "K"),
        ("P_net", "Kuma net giren güç (şarjda +, deşarjda −)", "W"),
        ("T_b", "Bekleme sonundaki kum sıcaklığı", "°C"),
    ]))
    st.markdown("**Sistem gücü, süre ve verim**")
    st.markdown(md_table([
        ("P_h", "Isıtıcı rezistans nominal elektrik gücü", "MW"),
        ("P_d", "Eşanjör deşarj (ısıl) gücü", "MW"),
        ("η_h", "Isıtıcı verimi (elektrik → ısı)", "–"),
        ("η_RT", "Round-trip verim, verilen ısı / çekilen elektrik", "% (–)"),
        ("t_şarj, t_deşarj", "Çevrimdeki şarj ve gerçek (kayıplı) deşarj süreleri", "saat"),
        ("t_bekleme", "Şarj sonu ile deşarj başı arasındaki bekleme süresi", "gün"),
        ("E_in, E_out", "Şebekeden çekilen elektrik ve kullanıcıya verilen ısı", "MWh"),
    ]))
    st.markdown("**1-D termoklin modeli**")
    st.markdown(md_table([
        ("z", "Eksenel koordinat (silo üstünden aşağı)", "m"),
        ("N, Δz", "Eksenel hücre sayısı ve hücre kalınlığı (Δz = H/N)", "–, m"),
        ("Ċ", "Isı taşıyıcı akışkanın kapasite hızı, ṁ·c_f", "W/K"),
        ("k_eff", "Efektif eksenel iletkenlik (kum iletimi + dağılma)", "W/(m·K)"),
        ("T_in, T_çıkış", "Akışkanın yatağa giriş ve çıkış sıcaklığı", "°C"),
        ("UA_yan", "Yan yüzeyin ısıl iletkenliği (hücrelere eşit dağıtılır)", "W/K"),
        ("Kapasite kullanımı", "Deşarjda çekilen ısı / nominal Q", "%"),
        ("Termoklin kalınlığı", "%50 deşarjda ΔT'nin %90–%10 seviyeleri arası mesafe", "m"),
        ("Kesme eşiği", "Şarj/deşarjı bitiren çıkış sıcaklığı seviyesi (ΔT yüzdesi olarak)", "%"),
    ]))

    st.markdown("**Hibrit yatak, CSP ve modüler tesis**")
    st.markdown(md_table([
        ("φ₁, φ₂", "Malzeme 1 ve 2'nin hacim kesirleri (φ₁ + φ₂ = 1)", "–"),
        ("w₁, w₂", "Malzeme 1 ve 2'nin kütle kesirleri", "–"),
        ("ρ_k", "Kompozit yığın yoğunluğu", "kg/m³"),
        ("c_p,k", "Kompozit özgül ısı (kütle ağırlıklı ortalama)", "J/(kg·K)"),
        ("A_opt", "Toplam optik toplama alanı (Fresnel/helyostat)", "m²"),
        ("DNI", "Doğrudan normal ışınım", "W/m²"),
        ("η_opt", "Toplam optik verim", "–"),
        ("Q_solar", "CSP ile hesaplanan şarj (ısıl) gücü", "MW"),
        ("N", "Silo modül sayısı (3D sekmesi; optimizasyonda ekonomik ömür de N_yıl ile gösterilir)", "adet"),
        ("V/N", "Modül başına hacim", "m³"),
        ("Δ%", "Tek dev siloya göre toplam ısı kaybı artışı", "%"),
    ]))
    st.markdown("**Analiz ve optimizasyon**")
    st.markdown(md_table([
        ("V*", "Hedef enerjiyi sağlayan gerekli kum hacmi", "m³"),
        ("Q_hedef", "Kullanıcının istediği depolanan enerji", "MWh"),
        ("f", "Amaç fonksiyonu (ısı kaybı veya toplam maliyet)", "kW veya €"),
        ("c_ins", "Yalıtım malzemesi birim hacim maliyeti", "€/m³"),
        ("V_ins", "Yalıtım malzemesi hacmi", "m³"),
        ("c_E", "Kayıp ısının birim enerji değeri", "€/MWh"),
        ("E_kayıp,yıl", "Ortalama sıcaklıkta yıllık ısı kaybı enerjisi", "MWh/yıl"),
        ("T̄", "Çevrim ortalama kum sıcaklığı, (T_max + T_min)/2", "°C"),
        ("N", "Ekonomik ömür", "yıl"),
    ]))

    st.markdown("### Model varsayımları ve sınırlılıkları")
    st.markdown(
        """
- **Ana modeller:** 0-D toplu kapasite (hızlı, metrik kartları ve soğuma eğrisi) ve 1-D termoklin (katmanlaşma, kapasite kullanımı, gerçekçi verim).
- **c_p(T) lineer:** Kuvarsın 573 °C'deki α→β faz geçişi (c_p sıçraması) modellenmemiştir; T_max bu değerin üzerindeyse enerji hafif eksik tahmin edilebilir.
- **1-D model LTE varsayar:** Akışkan ve kum aynı yerel sıcaklıktadır (küçük tane, yüksek hacimsel ısı transfer katsayısı). Parçacık içi iletim ve akışkan–katı sıcaklık farkı (Schumann) ihmal edilmiştir.
- **Akış tanımı:** Ċ, nominal gücü ΔT = T_max − T_min üzerinden taşıyacak şekilde seçilir; ısıtıcı yükü şarj boyunca modüle olur, deşarj gücü çıkış sıcaklığı düştükçe nominalin altına iner.
- **k_eff ve kesme eşikleri** tasarımcı tercihidir ve termoklin kalınlığını/kapasite kullanımını doğrudan etkiler; ölçüm veya literatürle kalibre edilmelidir.
- **Isı kaybı yolu:** Yalnızca yalıtım + dış film direnci; kapaklar aynı yalıtım kalınlığındadır, zemine iletim ayrıca modellenmemiştir. k_yalıtım sabit alınır.
- **Round-trip verimi:** Elektrik → ısı → ısı (power-to-heat). Isı tekrar elektriğe çevrilecekse güç çevrimi verimiyle (örn. ≈ %20–40) çarpılmalıdır.
- **0-D deşarj süresi kartı** kayıpsız idealdir; kayıplı süre çevrim özetinde ve 1-D sekmesinde bulunur.
- **Optimizasyon:** Hacim, hedef enerjiden doğrudan çıkar; ızgara taraması + yakınlaştırma ile H/R ve t aranır. Maliyet amacı basit (iskontosuz) bir ön tasarım modelidir; birim fiyatlar kullanıcı girdisidir ve yıllık kayıp ortalama sıcaklıkta sürekli bekleme varsayar.
- **Hibrit yatak:** İdeal karışım (hacimler toplanır; ince tanelerin boşluk doldurarak yığın yoğunluğunu artırması ihmal edilir). Karışım homojen kabul edilir; 3D sekmedeki katmanlı görünüm yalnızca görsel temsildir.
- **CSP şarjı:** Q_solar sabit (sabit DNI, gündüz-gece/bulut kesintisi yok) ve tamamen kuma aktarılır (η_ısıtıcı uygulanmaz). Gerçek takvim süresi için günlük güneşlenme profili gerekir.
- **Modüler tesis:** N özdeş modül, aynı H/R ve yalıtım; güç eşit paylaştırılır, modüller arası ısıl etkileşim ve boru/kolektör kayıpları ihmal edilir. 1-D çözümde tek modül çözülüp enerjiler N ile çarpılır.
- **Parametre taraması** 0-D model üzerinde ince ızgarada, isteğe bağlı olarak 1-D modelde kaba ızgarada yapılır.
- Ön tasarım aracıdır; ayrıntılı tasarım için CFD veya iki-fazlı (Schumann) yatak modelleri gerekir.
        """
    )

# ------------------------- VERİ ----------------------------------------------
with tab_data:
    for i, s in enumerate(systems):
        c = s["cfg"]
        h_hi = float(enthalpy(c["t_max"], s["c0"], s["beta"]))
        h_lo = float(enthalpy(c["t_min"], s["c0"], s["beta"]))
        h_t = enthalpy(s["temp"], s["c0"], s["beta"])
        df = pd.DataFrame({
            "Zaman (gün)": t_axis / SEC_PER_DAY,
            "Zaman (ay)": months_axis,
            "Kum sıcaklığı (°C)": s["temp"],
            "Isı kaybı (kW)": s["loss"],
            "Kalan faydalı enerji (%)": np.clip((h_t - h_lo) / (h_hi - h_lo), 0, None) * 100.0,
            # Enerji korunumu: depodan çıkan toplam ısı = m [h(T_max) − h(T(t))]
            "Kümülatif ısı kaybı (MWh)": s["mass"] * (h_hi - h_t) / J_PER_MWH,
        })
        if s["loss_1d"] is not None:
            df["Isı kaybı 1-D (kW)"] = s["loss_1d"]
        if compare:
            st.markdown(f'<span class="sys-badge">{"🅰️" if i == 0 else "🅱️"} {s["name"]}</span>', unsafe_allow_html=True)
        if s["cycle"] is not None:
            cy, tc = s["cycle"], tcs[i]
            st.markdown("**Çevrim özeti (0-D | 1-D)**")
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Şarj süresi", f"{cy['t_charge_h']:,.1f} sa", delta=f"1-D: {tc['t_charge_h']:,.1f} sa", delta_color="off")
            q2.metric("Şarj girdi enerjisi", f"{cy['e_in_mwh']:,.1f} MWh", delta=f"1-D: {tc['e_in_mwh']:,.1f} MWh", delta_color="off")
            q3.metric("Gerçek deşarj süresi", f"{cy['t_discharge_h']:,.1f} sa", delta=f"1-D: {tc['t_dis_h']:,.1f} sa", delta_color="off")
            q4.metric("Verilen ısı", f"{cy['e_out_mwh']:,.1f} MWh", delta=f"1-D: {tc['e_out_mwh']:,.1f} MWh", delta_color="off")
        st.markdown("**Soğuma simülasyonu verisi**")
        wide_call(st.dataframe, df.round(3))
        st.download_button("⬇️ CSV olarak indir", data=df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"kum_bataryasi_soguma_{i + 1}.csv", mime="text/csv", key=f"dl_{i}")
