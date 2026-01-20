import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import requests
from io import StringIO
from scipy.optimize import brentq
from scipy.stats import poisson
import os

# ==========================================
# 🔧 基础配置
# ==========================================
st.set_page_config(
    page_title="⚽ 足球量化黑科技 V14",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings("ignore")

# 🌍 联赛配置
LEAGUE_CONFIG = {
    "英超 (Premier League)": ["E0", False], "英冠 (Championship)": ["E1", False],
    "英甲 (League One)": ["E2", False], "英乙 (League Two)": ["E3", False],
    "德甲 (Bundesliga)": ["D1", False], "德乙 (Bundesliga 2)": ["D2", False],
    "西甲 (La Liga)": ["SP1", False], "西乙 (Segunda)": ["SP2", False],
    "意甲 (Serie A)": ["I1", False], "意乙 (Serie B)": ["I2", False],
    "法甲 (Ligue 1)": ["F1", False], "法乙 (Ligue 2)": ["F2", False],
    "荷甲 (Eredivisie)": ["N1", False], "葡超 (Liga I)": ["P1", False],
    "土超 (Super Lig)": ["T1", False], "比甲 (Jupiler)": ["B1", False],
    "希超 (Ethniki)": ["G1", False], "苏超 (Premiership)": ["SC0", False],
    "巴西甲 (Brazil Serie A)": ["BRA", True], "阿根廷甲 (Argentina)": ["ARG", True],
    "美职联 (USA MLS)": ["USA", True], "墨超 (Mexico Liga MX)": ["MEX", True],
    "日职联 (Japan J-League)": ["JPN", True], "中超 (China Super Lg)": ["CHN", True],
    "瑞典超 (Allsvenskan)": ["SWE", True], "挪威超 (Eliteserien)": ["NOR", True],
    "芬兰超 (Veikkausliiga)": ["FIN", True], "瑞士超 (Super League)": ["SWZ", True],
    "爱尔兰超 (Ireland)": ["IRL", True], "波兰甲 (Ekstraklasa)": ["POL", True],
    "罗甲 (Romania Liga 1)": ["ROU", True], "俄超 (Russia Premier)": ["RUS", True]
}

# 球队名修正
TEAM_NAME_FIX = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Tottenham Hotspur": "Tottenham", "Wolverhampton Wanderers": "Wolves",
    "Oxford United": "Oxford", "Charlton Athletic": "Charlton",
    "Botafogo RJ": "Botafogo", "Flamengo RJ": "Flamengo",
    "Atletico MG": "Atletico-MG", "Paranaense": "Athletico-PR",
    "River Plate": "River Plate", "Boca Juniors": "Boca Juniors",
    "Inter Miami": "Inter Miami CF"
}


# ==========================================
# 🧠 数学内核
# ==========================================
class PoissonEngine:
    @staticmethod
    def prob_over_2_5_to_lambda(prob_over_2_5):
        def error_func(lam):
            return (1 - poisson.cdf(2, lam)) - prob_over_2_5

        try:
            return brentq(error_func, 0.1, 10)
        except:
            return None

    @staticmethod
    def calculate_prob_for_line(lam, line):
        if line % 0.5 == 0 or line % 1 == 0: return 1 - poisson.cdf(int(line), lam)
        lower, upper = line - 0.25, line + 0.25
        return (1 - poisson.cdf(int(lower), lam) + 1 - poisson.cdf(int(upper), lam)) / 2


# ==========================================
# 🧠 后端引擎 (适配 Streamlit 缓存)
# ==========================================
class HybridBackend:
    def __init__(self):
        self.main_url = "https://www.football-data.co.uk/mmz4281/{}/{}.csv"
        self.extra_url = "https://www.football-data.co.uk/new/{}.csv"
        self.seasons = ['2122', '2223', '2324', '2425']

    # 使用 Streamlit 的缓存装饰器，防止每次点击按钮都重新下载
    @st.cache_data(ttl=3600)
    def download_data(_self, code, is_extra):
        # 注意：Streamlit云端没有本地存储的概念，我们尽量直接用内存
        try:
            all_dfs = []
            if not is_extra:
                for s in _self.seasons:
                    url = _self.main_url.format(s, code)
                    try:
                        r = requests.get(url, timeout=5)
                        if r.status_code == 200:
                            all_dfs.append(pd.read_csv(StringIO(r.text)))
                    except:
                        pass
            else:
                url = _self.extra_url.format(code)
                try:
                    r = requests.get(url, timeout=10)
                    if r.status_code == 200:
                        all_dfs.append(pd.read_csv(StringIO(r.text)))
                except:
                    pass

            if all_dfs:
                full = pd.concat(all_dfs, ignore_index=True)
                full.columns = [c.strip() for c in full.columns]
                return full
        except Exception:
            return None
        return None

    def process_league(self, code, is_extra):
        df = self.download_data(code, is_extra)
        if df is None or len(df) < 50: return None, None, None, False

        has_shots = 'HST' in df.columns
        clean_df = self.feature_engineering(df, has_shots)
        if clean_df.empty: return None, None, None, False

        model = self.train_model(clean_df, has_shots)
        stats = self.get_latest_stats(clean_df, has_shots)
        return model, stats, clean_df, has_shots

    def feature_engineering(self, df, has_shots):
        cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365>2.5', 'Avg>2.5']
        if has_shots: cols.extend(['HST', 'AST'])

        # 宽松模式：只要有基本列就行
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols].copy().dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])

        if 'B365>2.5' not in df.columns:
            if 'Avg>2.5' in df.columns:
                df['B365>2.5'] = df['Avg>2.5']
            else:
                return pd.DataFrame()

        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df.sort_values('Date', inplace=True)
        df['IsOver'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)

        def get_rolling(ts, vs):
            return ts.groupby(ts).transform(lambda x: vs.shift(1).rolling(3).mean())

        df['H_G_MA3'] = get_rolling(df['HomeTeam'], df['FTHG'])
        df['A_G_MA3'] = get_rolling(df['AwayTeam'], df['FTAG'])
        df['H_C_MA3'] = get_rolling(df['HomeTeam'], df['FTAG'])
        df['A_C_MA3'] = get_rolling(df['AwayTeam'], df['FTHG'])

        # ELO
        elo_dict = {t: 1500 for t in set(df['HomeTeam']).union(set(df['AwayTeam']))}
        h_elos, a_elos = [], []
        k = 20
        for idx, row in df.iterrows():
            h, a = row['HomeTeam'], row['AwayTeam']
            res = 1 if row['FTHG'] > row['FTAG'] else (0.5 if row['FTHG'] == row['FTAG'] else 0)
            exp = 1 / (1 + 10 ** ((elo_dict[a] - elo_dict[h]) / 400))
            h_elos.append(elo_dict[h]);
            a_elos.append(elo_dict[a])
            elo_dict[h] += k * (res - exp);
            elo_dict[a] += k * ((1 - res) - (1 - exp))
        df['ELO_Diff'] = abs(np.array(h_elos) - np.array(a_elos))

        if has_shots:
            df['H_S_MA3'] = get_rolling(df['HomeTeam'], df['HST'])
            df['A_S_MA3'] = get_rolling(df['AwayTeam'], df['AST'])
            df['H_Eff'] = df['FTHG'] / (df['HST'] + 0.1)
            df['A_Eff'] = df['FTAG'] / (df['AST'] + 0.1)
            df['H_Eff_MA3'] = get_rolling(df['HomeTeam'], df['H_Eff'])
            df['A_Eff_MA3'] = get_rolling(df['AwayTeam'], df['A_Eff'])

        return df.dropna()

    def train_model(self, df, has_shots):
        cols = ['H_G_MA3', 'A_G_MA3', 'H_C_MA3', 'A_C_MA3', 'ELO_Diff']
        if has_shots: cols.extend(['H_S_MA3', 'A_S_MA3', 'H_Eff_MA3', 'A_Eff_MA3'])

        X = df[cols];
        y = df['IsOver']
        model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.01, max_depth=4)
        model.fit(X, y)
        return model

    def get_latest_stats(self, df, has_shots):
        latest = {}
        teams = set(df['HomeTeam']).union(set(df['AwayTeam']))

        elo_dict = {t: 1500 for t in teams}
        k = 20
        for idx, row in df.iterrows():
            h, a = row['HomeTeam'], row['AwayTeam']
            res = 1 if row['FTHG'] > row['FTAG'] else (0.5 if row['FTHG'] == row['FTAG'] else 0)
            exp = 1 / (1 + 10 ** ((elo_dict[a] - elo_dict[h]) / 400))
            elo_dict[h] += k * (res - exp);
            elo_dict[a] += k * ((1 - res) - (1 - exp))

        for t in teams:
            matches = df[(df['HomeTeam'] == t) | (df['AwayTeam'] == t)]
            if matches.empty: continue
            last = matches.iloc[-1]
            p = 'H' if last['HomeTeam'] == t else 'A'
            stats = {'G': last[f'{p}_G_MA3'], 'C': last[f'{p}_C_MA3'], 'ELO': elo_dict[t]}
            if has_shots:
                stats['S'] = last[f'{p}_S_MA3']
                stats['Eff'] = last[f'{p}_Eff_MA3']
            latest[t] = stats
        return latest

    def predict(self, model, stats, h, a, odds, line, has_shots):
        h_fix, a_fix = TEAM_NAME_FIX.get(h, h), TEAM_NAME_FIX.get(a, a)
        if h_fix not in stats or a_fix not in stats: return None

        hs, as_ = stats[h_fix], stats[a_fix]
        elo_diff = abs(hs['ELO'] - as_['ELO'])

        if has_shots:
            feat = np.array(
                [[hs['G'], as_['G'], hs['C'], as_['C'], elo_diff, hs['S'], as_['S'], hs['Eff'], as_['Eff']]])
        else:
            feat = np.array([[hs['G'], as_['G'], hs['C'], as_['C'], elo_diff]])

        base_prob = model.predict_proba(feat)[0][1]
        lam = PoissonEngine.prob_over_2_5_to_lambda(base_prob)
        final_prob = PoissonEngine.calculate_prob_for_line(lam, line) if lam else base_prob
        ev = (final_prob * odds) - 1
        return base_prob, final_prob, ev


# ==========================================
# 🖥️ 前端：Streamlit 页面逻辑
# ==========================================
backend = HybridBackend()

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 联赛选择")
    selected_league_name = st.selectbox("请选择联赛:", list(LEAGUE_CONFIG.keys()))

    # 获取联赛配置
    code, is_extra = LEAGUE_CONFIG[selected_league_name]

    st.info("数据将实时从英国服务器获取，首次加载可能需要几秒钟。")

    # 加载数据 (带 Spinner)
    with st.spinner(f"正在下载并训练 {selected_league_name} AI模型..."):
        model, stats, clean_df, has_shots = backend.process_league(code, is_extra)

    if model:
        st.success(f"✅ {selected_league_name} 就绪")
        engine_type = "🚀 精密引擎 (含射正)" if has_shots else "🍃 轻量引擎 (无射正)"
        st.caption(f"当前模式: {engine_type}")

        # 球队列表展示
        team_list = sorted(stats.keys())
        with st.expander("查看该联赛球队列表"):
            st.write(team_list)
    else:
        st.error("❌ 数据加载失败，请检查网络或更换联赛")

# --- 主页面 ---
st.title("⚽ 足球量化 AI 终端 V14 (网页版)")
st.markdown("---")

if model and stats:
    tab1, tab2 = st.tabs(["📊 批量扫盘 (Batch)", "🔮 单场预测 (Single)"])

    # --- Tab 1: 批量 ---
    with tab1:
        st.subheader("批量分析")
        st.caption("输入格式：主队,客队,赔率,盘口 (每行一场)")
        default_text = "Man City,Liverpool,1.85,3.0\nArsenal,Chelsea,2.00,2.5"
        input_text = st.text_area("在此粘贴比赛:", value=default_text, height=150)

        if st.button("🚀 开始分析", type="primary"):
            lines = input_text.strip().split('\n')
            results = []

            for line_str in lines:
                if ',' not in line_str: continue
                try:
                    parts = [p.strip() for p in line_str.split(',')]
                    h, a, o = parts[0], parts[1], float(parts[2])
                    l = float(parts[3]) if len(parts) > 3 else 2.5

                    res = backend.predict(model, stats, h, a, o, l, has_shots)
                    if res:
                        bp, fp, ev = res
                        rec = "🔥重注" if ev > 0.05 and fp > 0.55 else ("💰买入" if ev > 0 else "❄️放弃")
                        results.append({
                            "比赛": f"{h} vs {a}",
                            "盘口": l,
                            "真实概率": f"{fp:.1%}",
                            "EV值": ev,
                            "建议": rec
                        })
                except:
                    pass

            if results:
                # 转为 DataFrame 展示
                res_df = pd.DataFrame(results)
                res_df = res_df.sort_values("EV值", ascending=False)


                # 样式高亮函数
                def highlight_row(row):
                    if "重注" in row['建议']: return ['background-color: #ffcccc'] * len(row)
                    if "买入" in row['建议']: return ['background-color: #dff9fb'] * len(row)
                    return [''] * len(row)


                st.dataframe(res_df.style.apply(highlight_row, axis=1), use_container_width=True)
            else:
                st.warning("未检测到有效输入或球队名拼写错误")

    # --- Tab 2: 单场 ---
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            team_list = sorted(stats.keys())
            home = st.selectbox("主队", team_list)
            away = st.selectbox("客队", team_list, index=1)
        with col2:
            odds = st.number_input("大球赔率", value=1.90, step=0.01)
            line = st.selectbox("盘口", [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5], index=2)

        if st.button("开始预测"):
            res = backend.predict(model, stats, home, away, odds, line, has_shots)
            if res:
                bp, fp, ev = res

                # 结果展示卡片
                st.markdown("### 🤖 AI 分析报告")
                c1, c2, c3 = st.columns(3)
                c1.metric("基础大球指数", f"{bp:.1%}")
                c2.metric(f"真实概率 (>{line})", f"{fp:.1%}")
                c3.metric("期望价值 (EV)", f"{ev:.2f}", delta_color="normal" if ev > 0 else "inverse")

                if ev > 0.05:
                    st.success(f"🔥 **强烈推荐：重注大球！** (EV = {ev:.2f})")
                elif ev > 0:
                    st.info(f"💰 **值得投资：轻注大球** (EV = {ev:.2f})")
                else:
                    st.error(f"❄️ **建议放弃** (EV 为负，无利可图)")
            else:
                st.error("数据不足")

else:
    st.info("👈 请在左侧选择联赛以开始...")