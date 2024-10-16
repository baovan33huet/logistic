from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.impute import SimpleImputer

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def helloWord():
    return '222'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Lấy dữ liệu từ query parameters
        gender = data.get('gender')
        seniorCitizen = data.get('seniorCitizen')
        partner = data.get('partner')
        dependents = data.get('dependents')
        tenure = data.get('tenure')  
        phoneService = data.get('phoneService') 	
        multipleLines = data.get('multipleLines')
        contract_Month_to_month = data.get('contract_Month_to_month')
        contract_One_year = data.get('contract_One_year')
        contract_Two_year = data.get('contract_Two_year')
        internetService_DSL = data.get('internetService_DSL')
        internetService_Fiber_optic = data.get('internetService_Fiber_optic')
        internetService_No = data.get('internetService_No')
        onlineSecurity = data.get('onlineSecurity')
        onlineBackup = data.get('onlineBackup')
        deviceProtection = data.get('deviceProtection')
        techSupport = data.get('techSupport')
        streamingTV = data.get('streamingTV')  
        streamingMovies = data.get('streamingMovies') 	
        paperlessBilling = data.get('paperlessBilling')
        paymentMethod_Bank_transfer = data.get('paymentMethod_Bank_transfer')   
        paymentMethod_Credit_card = data.get('paymentMethod_Credit_card')
        paymentMethod_Electronic_check = data.get('paymentMethod_Electronic_check') 
        paymentMethod_Mailed_check = data.get('paymentMethod_Mailed_check') 
        monthlyCharges = data.get('monthlyCharges')
        totalCharges = data.get('totalCharges')

        feature = np.array([
        gender, seniorCitizen, partner, dependents, tenure, phoneService,
        multipleLines, contract_Month_to_month, contract_One_year, contract_Two_year, internetService_DSL,
        internetService_Fiber_optic, internetService_No, onlineSecurity, onlineBackup, deviceProtection,
        techSupport,streamingTV, streamingMovies, paperlessBilling, paymentMethod_Bank_transfer, paymentMethod_Credit_card,
        paymentMethod_Electronic_check, paymentMethod_Mailed_check, monthlyCharges, totalCharges
        ])
        # feature_list = feature.tolist()


        feature_2d = feature.reshape(1, -1)
        imputer = SimpleImputer(strategy='mean')
        feature_2d_imputed = imputer.fit_transform(feature_2d)
        result = model.predict(feature_2d_imputed)
        return jsonify({'prediction': int(result)})
    except Exception as e:
        # In ra lỗi và trả về lỗi chi tiết để debug
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True, port=5001)