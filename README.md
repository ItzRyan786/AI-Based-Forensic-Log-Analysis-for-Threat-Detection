V Project Workflow 


Step 1: System Overview (Dashboard)
•	Launch the application, and you will land on the Dashboard by default. The system will load initial data (synthetic by default) to show the current security posture.
<img width="900" height="319" alt="image" src="https://github.com/user-attachments/assets/1912b70a-a921-4616-b42a-1e5bfbef7282" />



 
Step 2: Data Ingestion (Loading Data)
•	Go to the data ingestion Menu in the tab bar, select Real data. Here, you can upload log files, or you can select the synthetic data option, where the model itself would simulate a threat detection environment to give an overview to the user on how it detects threats.
 
<img width="900" height="402" alt="image" src="https://github.com/user-attachments/assets/3893be1f-cf18-4c40-b981-8b0f711c028f" />








Step 3: Model Training (The AI Brain)

•	Now go to the model training menu. By default, the isolate forest model will be selected. Make sure you train the model and wait for it to complete. The system will analyze the data and learn different patterns based on it to distinguish between normal and abnormal behaviours.

<img width="900" height="451" alt="image" src="https://github.com/user-attachments/assets/c11c42ca-fb67-468d-b538-48670006e39f" />



 



















Step 4: Explainable AI (Feature Importance)

•	On the same page, you can see a feature importance tab, where you can see the reason why it flags certain things.

(e.g.,"Volume_MB" is the biggest risk factor).
<img width="900" height="456" alt="image" src="https://github.com/user-attachments/assets/220337ef-59c8-439b-bb05-39814c201313" />



 



Step 5: Real-Time Monitoring (Live Detection)
Now go to the live detection menu in the tab bar. Click "Start Live Monitoring," and you will see some alerts in red or in orange colour. Here, it can be said that the system is simulating the threats as real-time traffic happens and flagging them.   
<img width="900" height="494" alt="image" src="https://github.com/user-attachments/assets/742d4756-71d7-4792-827b-e571c0fa8cda" />












Step 6: Incident Response (Managing Alerts)
•	 Then there is an Alert and cases tab here, where you can create a case where an analyst who is reviewing the threat can immediately classify the threat if manual intervention is required. 

<img width="900" height="488" alt="image" src="https://github.com/user-attachments/assets/36cbf3d7-aa8b-4ec9-9e4f-d2be77840cb1" />

 

Step 7: Reporting (Executive Summary)
•	Now there is also a reporting tab where you can select the Reporting and Export option to generate a Report. The system will gather all the evidence and make it into a report that can be viewed, and appropriate action can be taken.  
<img width="900" height="483" alt="image" src="https://github.com/user-attachments/assets/e92beab6-908d-4506-88fb-b5886624f9c6" />

Testing & Results
•	Simulation Environment:
Synthetic data was generated so that we can also simulate an environment with normal and malicious users without using real data. Tested Scenarios:
•	Insider data exfiltration attempt → CRITICAL
•	Off-hour privileged access → HIGH
•	 Multiple failed login attempts → MEDIUM
•	 Normal user activity baseline → LOW

Our system showed improved accuracy and reduced false positives compared to traditional models

