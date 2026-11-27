"""
Запуск:  streamlit run app.py
(FastAPI должен быть запущен: uvicorn main:app  или  Docker-контейнер)
"""
import os

import requests
import streamlit as st

# Адрес API. Через переменную окружения — чтобы в Docker можно было подменить.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Churn Prediction", page_icon="📡", layout="wide")


# (Next Best Offer)
def retention_offers(c: dict) -> list[str]:
    """Простые бизнес-правила: какое предложение сделать клиенту."""
    offers = []
    if c["Contract"] == "Month-to-month":
        offers.append("Предложить годовой контракт со скидкой 15–20%")
    if c["InternetService"] == "Fiber optic" and c["TechSupport"] == "No":
        offers.append("Подключить техподдержку бесплатно на 3 месяца")
    if c["InternetService"] != "No" and c["OnlineSecurity"] == "No":
        offers.append("Дать пакет «Онлайн-защита» в подарок на 2 месяца")
    if c["PaymentMethod"] == "Electronic check":
        offers.append("Перевести на автоплатёж картой с бонусом 500 ₸")
    if c["tenure"] < 12:
        offers.append("Звонок менеджера: помочь новому клиенту освоиться")
    if c["MonthlyCharges"] > 80:
        offers.append("Подобрать более выгодный пакет услуг")
    return offers[:3]  


# Шапка
st.title("📡 Telco Customer Churn Prediction")
st.caption("Введите данные клиента — модель оценит риск ухода и предложит, как его удержать.")

# Проверяем, что API доступен (GET /)
try:
    requests.get(f"{API_URL}/", timeout=3).raise_for_status()
    st.success(f"API подключён: {API_URL}")
except requests.RequestException:
    st.error(f"API недоступен по адресу {API_URL}. Запустите FastAPI (uvicorn или Docker).")
    st.stop()

# Форма с данными клиента 
with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Клиент")
        gender = st.selectbox("Пол", ["Female", "Male"])
        senior = st.selectbox("Пенсионер (65+)", ["No", "Yes"])
        partner = st.selectbox("Есть партнёр", ["No", "Yes"])
        dependents = st.selectbox("Есть иждивенцы", ["No", "Yes"])
        tenure = st.slider("Сколько месяцев с нами (tenure)", 0, 72, 2)

    with col2:
        st.subheader("Услуги")
        phone = st.selectbox("Телефонная связь", ["Yes", "No"])
        multiple = st.selectbox("Несколько линий", ["No", "Yes"])
        internet = st.selectbox("Интернет", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Онлайн-защита", ["No", "Yes"])
        backup = st.selectbox("Онлайн-бэкап", ["No", "Yes"])
        protection = st.selectbox("Защита устройств", ["No", "Yes"])
        support = st.selectbox("Техподдержка", ["No", "Yes"])
        tv = st.selectbox("Стриминг ТВ", ["Yes", "No"])
        movies = st.selectbox("Стриминг фильмов", ["Yes", "No"])

    with col3:
        st.subheader("Договор и оплата")
        contract = st.selectbox("Тип контракта", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Электронные счета", ["Yes", "No"])
        payment = st.selectbox("Способ оплаты", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])
        monthly = st.number_input("Ежемесячный платёж, $", 0.0, 200.0, 95.5, step=0.5)

    submitted = st.form_submit_button("🔍 Оценить риск оттока", use_container_width=True)

# Отправка в API и вывод результата 
if submitted:
    no_inet = internet == "No"
    inet = lambda v: "No internet service" if no_inet else v  

    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": "No phone service" if phone == "No" else multiple,
        "InternetService": internet,
        "OnlineSecurity": inet(security),
        "OnlineBackup": inet(backup),
        "DeviceProtection": inet(protection),
        "TechSupport": inet(support),
        "StreamingTV": inet(tv),
        "StreamingMovies": inet(movies),
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": round(monthly * tenure, 2),
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=customer, timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as e:
        st.error(f"Ошибка при обращении к API: {e}")
        st.stop()

    prob = result["probability"]
    risk = result["risk"]

    st.divider()
    left, right = st.columns([1, 2])

    with left:
        st.metric("Вероятность ухода", f"{prob:.0%}")
        st.progress(prob)
        if risk == "high":
            st.error("🔴 Высокий риск оттока")
        elif risk == "medium":
            st.warning("🟡 Средний риск оттока")
        else:
            st.success("🟢 Низкий риск оттока")

    with right:
        st.subheader("🎯 Рекомендации по удержанию (Next Best Offer)")
        if risk == "low":
            st.write("Клиент лоялен — специальные действия не нужны. "
                     "Достаточно стандартных коммуникаций.")
        else:
            offers = retention_offers(customer)
            if offers:
                for offer in offers:
                    st.write(f"- {offer}")
            else:
                st.write("Персональное предложение: связаться с клиентом и узнать причины недовольства.")

    with st.expander("Что было отправлено в API (JSON)"):
        st.json(customer)
