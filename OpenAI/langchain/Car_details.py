import streamlit as st
import prompt_gen

st.title("Car Details")

body_type=st.sidebar.selectbox("Pick vehicle the Body-type:",("NONE","SEDAN","HATCHBACK","SUV","MUV"))

price_range = st.sidebar.selectbox("Pick Budget:",("NONE","Below 10 lakhs","Between 10 lakhs to 50 lakhs","Above 50 lakhs"))

car_name=None


if body_type !="NONE" and price_range != "NONE":
    car_list=prompt_gen.Car_list(body_type,price_range).strip(",")
    car_name = st.sidebar.selectbox("Select any Car",tuple(car_list.split(",")))

    st.warning("Please select any vehicles!!!")
else:
    st.warning("Please select the fields!!!")


if car_name:
    res=prompt_gen.generate_details(car_name)
    st.write(res["Description_car"])
    st.write(res["main_features"])
    st.write("THANK YOU..")
    