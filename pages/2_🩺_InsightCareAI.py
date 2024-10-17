import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from Bio import Entrez, Medline
import openai
from openai import OpenAI

# Add dynamic background color change with grey-to-blue gradient and subtle animation
from streamlit_lottie import st_lottie
import json

# Define the gradient background and top bar with the updated title
from PIL import Image
import streamlit as st
from PIL import Image
import base64


import requests
import re
import difflib


import streamlit.components.v1 as components


# Set page configuration
st.set_page_config(
    page_title="InsightCare.AI",
    page_icon="💡",
    layout="wide"
)









# Load the provided CSV files
df_antibiotics = pd.read_csv('cleaned_antibiotics.csv')
df_variables = pd.read_csv('Variables.csv')
df_aware = pd.read_csv('WHO_AWARE.csv')

# Extract unique values for the dropdowns from the Variables.csv
unique_antibiotics = df_variables['Antibiotics'].unique()
unique_antibiotics_prev = df_variables['Antibiotics_prev'].unique()
unique_countries = df_variables['Country'].unique()
unique_phenotypes = df_variables['Phenotype'].unique()
unique_sources = df_variables['Source'].unique()
unique_specialities = df_variables['Speciality'].unique()
unique_diagnosis = df_variables['Infection_Diagnosis'].unique()

# Set your email for the Entrez API (required by NCBI)
Entrez.email = "fredmutisya11@gmail.com"



# Load the Lottie animation
with open("lottie/emr_blue.json", "r") as f:
    emr_sidebar = json.load(f)





# Define the gradient background and top bar with the button transition effect
st.markdown(
    """
    <style>
    @keyframes gradientBackground {
        0% {background-color: white;}
        50% {background-color: #6fc2f2;}
        100% {background-color: #6fc2f2;}
    }

    .stApp {
        background: linear-gradient(135deg, white 0%, #FAF9F6 100%);
        animation: gradientBackground 5s infinite alternate;
    }

    .top-bar {
        background: linear-gradient(135deg, #24C6DC, #106EBE);
        padding: 15px;
        color: #F0F4F8;
        font-size: 84px;
        text-align: center;
        border-bottom: 3px solid #0FFCBE;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        font-family: 'Roboto', sans-serif;
        font-weight: bold;
        width: 100%;
        position: absolute;
        top: 0;
        left: 0;
        z-index: 1000;
    }

    .lottie-container {
        width: 100px;
        height: 100px;
        margin: 0 auto;
    }

    .stButton>button {
        background: linear-gradient(135deg, #24C6DC, #e800e4);
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        transition: background 0.3s ease-in-out;
        cursor: pointer;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg,  #e800e4, #24C6DC);
        color: #F0F4F8;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Load the logo image (if needed)
logo_path = 'logos/logo-color.png'
logo_image = Image.open(logo_path)
st.image(logo_image, use_column_width=True)








# Sidebar with Lottie animation
with st.sidebar:
    st_lottie(emr_sidebar, height=150, width=150, key="sidebar-animation")








# Sample data for three patients with placeholder images
patients = {
    'Patient ID': ['P002', 'P001', 'P003'],  # Switched order of P001 and P002
    'Name': ['Jane Wanjiku', 'Chris Walters', 'Zawadi Hassan'],  # Switched names
    'Age': [34, 82, 2],  # Switched ages
    'Gender': ['Female', 'Male', 'Male'],  # Switched genders
    'Diagnosis': ['Pneumonia', 'Infectious Endocarditis', 'Meningitis'],  # Switched diagnoses
    'Medications': ['Amoxicillin', 'Ceftriaxone', 'Ceftazidime'],  # Switched medications
    'Last Visit': ['2024-07-23', '2024-08-01', '2024-08-10'],  # Switched last visit dates
    'Image': [
        'images/young_african_female.jpg',  # Switched images
        'images/old_white_male.jpg',
        'images/young_boy.jpg'
    ]
}


# Convert the dictionary into a DataFrame
df_patients = pd.DataFrame(patients)

# Function to display patient details
def display_patient_details(patient_id):
    st.subheader("Patient Details")
    patient_data = df_patients[df_patients['Patient ID'] == patient_id]
    if not patient_data.empty:
        st.image(patient_data.iloc[0]['Image'], width=150)
        st.write("**Name:**", patient_data.iloc[0]['Name'])
        st.write("**Age:**", patient_data.iloc[0]['Age'])
        st.write("**Gender:**", patient_data.iloc[0]['Gender'])
        st.write("**Diagnosis:**", patient_data.iloc[0]['Diagnosis'])
        st.write("**Medications:**", patient_data.iloc[0]['Medications'])
        st.write("**Last Visit:**", patient_data.iloc[0]['Last Visit'])
    else:
        st.error("Patient not found")

# Check if a specific patient is selected via query params
query_params = st.query_params
if 'patient_id' in query_params:
    selected_patient_id = query_params['patient_id'][0]
    display_patient_details(selected_patient_id)
    st.button("Back to Main Page", on_click=lambda: st.experimental_set_query_params())
else:
    # Sidebar for displaying patients with images and clickable names
    st.sidebar.header("Patient List")
    for i in range(len(df_patients)):
        col1, col2 = st.sidebar.columns([1, 3])
        with col1:
            if st.sidebar.button("", key=f"image-{i}"):
                pass
        with col2:
            if st.sidebar.button(df_patients.iloc[i]['Name'], key=f"name-{i}"):
                pass
        st.sidebar.image(df_patients.iloc[i]['Image'], width=100)

    # Main area for adding new patient information
    st.subheader("Add a New Patient")

    new_id = st.text_input("Patient ID", placeholder='HA004')
    new_name = st.text_input("Name", placeholder='John Doe')


    # New fields for Antibiogram with dropdowns populated from Variables.csv

    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox("Residence", unique_countries)
        age_group = st.selectbox("Age Group", ['0-18', '19-35', '36-50', '51+'])

    with col2:
        gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
        speciality = st.selectbox("Speciality", unique_specialities)

    # New large text box for History and Physical Exam Findings
    history_and_physical = st.text_area("History and Physical Exam Findings", placeholder='History: The patient is a 65-year-old male with a 2-week history of progressive shortness of breath, fatigue, and occasional chest tightness. He reports a productive cough with yellow sputum and denies any fever, chills, or recent travel. His medical history includes hypertension and hyperlipidemia. He is a former smoker with a 30-pack-year history and quit 5 years ago.Physical Exam:Vital Signs: BP 140/90 mmHg, HR 95 bpm, RR 18/min, SpO2 92 on room air, Temp 37.2°C.General: Alert and oriented, in mild respiratory distress.Respiratory: Dullness to percussion and decreased breath sounds over the right lower lung field, scattered crackles on auscultation. Cardiovascular: Regular rhythm, no murmurs, rubs, or gallops.Abdomen: Soft, non-tender, no masses, normal bowel sounds.Extremities: No cyanosis, clubbing, or edema.', height=200)

    # Smaller boxes for Laboratory Tests and Imaging
    lab_tests = st.text_input("Laboratory Tests", placeholder='CBC: WBC 12.5 x 10^9/L (Neutrophilia), Hb 13.5 g/dL, Platelets 250 x 10^9/L.CRP: 85 mg/L (Elevated).Electrolytes: Sodium 138 mmol/L, Potassium 4.1 mmol/L, Chloride 100 mmol/L, Bicarbonate 24 mmol/L.Liver Function Tests: AST 45 U/L, ALT 30 U/L, Total Bilirubin 1.0 mg/dL.Renal Function: BUN 18 mg/dL, Creatinine 1.2 mg/dL, GFR 70 mL/min/1.73 m².')
    imaging = st.text_input("Imaging Results", placeholder ='Chest X-ray: Patchy consolidation in the right lower lobe, consistent with pneumonia. No pleural effusion or pneumothorax.CT Chest: Confirmed right lower lobe consolidation with associated air bronchograms. No evidence of pulmonary embolism.')



    col3, col4 = st.columns(2)

    with col3:
        source = st.selectbox("Source", unique_sources)
        infection_diagnosis = st.selectbox("Diagnosis", unique_diagnosis)
        
    with col4:
        antibiotic_prev = st.selectbox("Previous Antibiotic Used in current Illness", unique_antibiotics)
        antibiotic = st.selectbox("Antibiotic", unique_antibiotics)
        
    antibiotic_dose = st.text_input("Antibiotic route and dose")
    
    import streamlit as st

    # Additional tabs for consulting resources and generating reports
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Ask InsightCare.AI",
        "🩺 Medication Info", 
        "🧪 Laboratory Info", 
        "🔬 Consult Research"
    ])

    # Increase font size of the tab headers
    css = '''
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size:1.5rem;
        }
    </style>
    '''

    st.markdown(css, unsafe_allow_html=True)










    with tab2:
        # Load the CSV file containing antibiotic data
        df_aware = pd.read_csv('WHO_AWARE.csv', encoding='ISO-8859-1')
        
        # Find the row in the DataFrame that matches the selected antibiotic
        selected_row = df_aware[df_aware['Antibiotic'] == antibiotic].iloc[0]
        
        # Extract relevant details from the selected row
        antibiotic_class = selected_row['Class']
        category = selected_row['Category']

        # Bullet points for FDA information
        st.markdown("### AWARE Information:")
        # Construct the output message
        output_message = f"{antibiotic} is one of the {antibiotic_class} and is in the AWARE category {category}."
        
        # Display the message
        st.write(output_message)

        # Bullet points for FDA information
        st.markdown("### FDA Information:")
        
        # Regex function to format text as bullet points
        def format_as_bullets(text):
            bullet_points = re.sub(r"(^|\n)(-)", r"\n- ", text.strip())
            return bullet_points

        # Fetch Drug Labeling Information
        st.markdown("- **Drug Labeling**:")
        base_url_label = "https://api.fda.gov/drug/label.json"
        params_label = {
            "search": f"openfda.brand_name:{antibiotic.lower()}",
            "limit": 1
        }
        response_label = requests.get(base_url_label, params=params_label)
        if response_label.status_code == 200:
            label_data = response_label.json()
            if label_data['results']:
                label_output = (
                    f"- **Brand Name**: {label_data['results'][0].get('openfda', {}).get('brand_name', ['N/A'])[0]}\n"
                    f"- **Generic Name**: {label_data['results'][0].get('openfda', {}).get('generic_name', ['N/A'])[0]}\n"
                    f"- **Indications and Usage**: {label_data['results'][0].get('indications_and_usage', ['N/A'])[0]}\n"
                )
                st.markdown(format_as_bullets(label_output))
            else:
                st.markdown("- No drug labeling information available.")
        else:
            st.markdown(f"- Failed to retrieve drug labeling information. Status code: {response_label.status_code}")
        
        # Fetch Adverse Events Information
        st.markdown("- ** Top 10 Adverse Events listed in the FDA Surveillance**:")
        base_url_event = "https://api.fda.gov/drug/event.json"
        params_event = {
            "search": f"patient.drug.medicinalproduct:{antibiotic.lower()}",
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": 10
        }
        response_event = requests.get(base_url_event, params=params_event)
        if response_event.status_code == 200:
            event_data = response_event.json()
            if 'results' in event_data:
                event_output = "\n".join(
                    [f"- **{event['term']}** (Count: {event['count']})" for event in event_data['results']]
                )
                st.markdown(format_as_bullets(event_output))
            else:
                st.markdown("- No adverse events information available.")
        else:
            st.markdown(f"- Failed to retrieve adverse events information. Status code: {response_event.status_code}")
















    with tab3:
        st.write("This tab allows you to consult the Surveillance data and generate a detailed antibiogram based on the selected criteria.")
        
        if st.button("Generate Filtered AST Surveillance Report"):
            # Initial filtering based on all criteria
            filtered_df = df_antibiotics[
                (df_antibiotics['Country'] == country) &
                (df_antibiotics['Gender'] == gender) &
                (df_antibiotics['Age.Group'] == age_group) &
                (df_antibiotics['Speciality'] == speciality) &
                (df_antibiotics['Source'] == source) &
                (df_antibiotics['Antibiotic'] == antibiotic) &
                (df_antibiotics['Infection_Diagnosis'] == infection_diagnosis)
            ]

            # If no exact match, relax criteria step by step
            if filtered_df.empty:
                st.write("No exact match found. Showing best available match by relaxing criteria.")
                # Relax Infection Diagnosis
                filtered_df = df_antibiotics[
                    (df_antibiotics['Country'] == country) &
                    (df_antibiotics['Gender'] == gender) &
                    (df_antibiotics['Age.Group'] == age_group) &
                    (df_antibiotics['Speciality'] == speciality) &
                    (df_antibiotics['Source'] == source) &
                    (df_antibiotics['Antibiotic'] == antibiotic)
                ]
                if not filtered_df.empty:
                    st.write("Match found by ignoring Infection Diagnosis.")
                else:
                    # Relax Speciality
                    filtered_df = df_antibiotics[
                        (df_antibiotics['Country'] == country) &
                        (df_antibiotics['Gender'] == gender) &
                        (df_antibiotics['Age.Group'] == age_group) &
                        (df_antibiotics['Source'] == source) &
                        (df_antibiotics['Antibiotic'] == antibiotic)
                    ]
                    if not filtered_df.empty:
                        st.write("Match found by ignoring Infection Diagnosis and Speciality.")
                    else:
                        # Relax Source
                        filtered_df = df_antibiotics[
                            (df_antibiotics['Country'] == country) &
                            (df_antibiotics['Gender'] == gender) &
                            (df_antibiotics['Age.Group'] == age_group) &
                            (df_antibiotics['Antibiotic'] == antibiotic)
                        ]
                        if not filtered_df.empty:
                            st.write("Match found by ignoring Infection Diagnosis, Speciality, and Source.")
                        else:
                            # Relax Age Group
                            filtered_df = df_antibiotics[
                                (df_antibiotics['Country'] == country) &
                                (df_antibiotics['Gender'] == gender) &
                                (df_antibiotics['Antibiotic'] == antibiotic)
                            ]
                            if not filtered_df.empty:
                                st.write("Match found by ignoring Infection Diagnosis, Speciality, Source, and Age Group.")
                            else:
                                # Relax Gender
                                filtered_df = df_antibiotics[
                                    (df_antibiotics['Country'] == country) &
                                    (df_antibiotics['Antibiotic'] == antibiotic)
                                ]
                                if not filtered_df.empty:
                                    st.write("Match found by ignoring Infection Diagnosis, Speciality, Source, Age Group, and Gender.")
                                else:
                                    # Relax Country
                                    filtered_df = df_antibiotics[
                                        (df_antibiotics['Antibiotic'] == antibiotic)
                                    ]
                                    if not filtered_df.empty:
                                        st.write("Match found by ignoring all criteria except Antibiotic.")
                                    else:
                                        st.write("No matching data found even after relaxing all criteria.")

            if not filtered_df.empty:
                st.dataframe(filtered_df)

                # Count the occurrences of each Species
                species_counts = filtered_df['Species'].value_counts().head(10)

                st.subheader("Top 10 Most Common Species by Resistance Levels")

                # Create a figure with subplots for each Species's resistance levels
                fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
                axes = axes.flatten()

                resistance_levels = []

                for i, species in enumerate(species_counts.index):
                    resistance_data = filtered_df[filtered_df['Species'] == species]['Resistance'].value_counts(normalize=True) * 100
                    resistance_data.plot(kind='bar', ax=axes[i], title=f"{species} Resistance Levels")
                    axes[i].set_xlabel('Resistance Level')
                    axes[i].set_ylabel('Percentage')

                    # Calculate the resistance percentage
                    resistant_percentage = resistance_data.get('Resistant', 0)
                    resistance_levels.append((species, resistant_percentage))

                plt.tight_layout()
                st.pyplot(fig)

                # Create a descriptive paragraph
                summary = f"This is a summary of the surveillance data for {antibiotic} based on the variables selected: age group {age_group}, gender {gender}, country {country}, speciality {speciality}, and source {source}. "
                summary += "The surveillance results show that the top 10 most common bacteria are "
                summary += ", ".join([f"{species} with resistance levels of {percentage:.2f}%" for species, percentage in resistance_levels]) + "."

                # Check for Staphylococcus aureus and calculate MRSA percentage if applicable
                if 'Staphylococcus aureus' in [species for species, _ in resistance_levels]:
                    # Filter data for Staphylococcus aureus and Oxacillin resistance
                    staph_aureus_df = filtered_df[(filtered_df['Species'] == 'Staphylococcus aureus') & (filtered_df['Antibiotic'] == 'Oxacillin')]

                    if not staph_aureus_df.empty:
                        total_staph_aureus = len(staph_aureus_df)
                        mrsa_count = staph_aureus_df[staph_aureus_df['Resistance'] == 'Resistant'].shape[0]
                        mrsa_percentage = (mrsa_count / total_staph_aureus) * 100

                        # Add MRSA percentage to the summary
                        summary += f"\n\nAmong the *Staphylococcus aureus* isolates, {mrsa_percentage:.1f}% are methicillin-resistant *S. aureus* (MRSA) based on their resistance to Oxacillin."

                st.write(summary)
                # Store the summary in session state
                st.session_state['summary_tab1'] = summary












        if st.button("Generate Detailed Antibiogram"):
            # Filter data based on selected criteria (without 'Antibiotic')
            filtered_df = df_antibiotics[
                (df_antibiotics['Country'] == country) &
                (df_antibiotics['Gender'] == gender) &
                (df_antibiotics['Age.Group'] == age_group) &
                (df_antibiotics['Speciality'] == speciality) &
                (df_antibiotics['Source'] == source) &
                (df_antibiotics['Infection_Diagnosis'] == infection_diagnosis)
            ]

            # If no exact match, relax criteria step by step
            if filtered_df.empty:
                st.write("No exact match found. Showing best available match by relaxing criteria.")

                # Relax Infection Diagnosis
                filtered_df = df_antibiotics[
                    (df_antibiotics['Country'] == country) &
                    (df_antibiotics['Gender'] == gender) &
                    (df_antibiotics['Age.Group'] == age_group) &
                    (df_antibiotics['Speciality'] == speciality) &
                    (df_antibiotics['Source'] == source)
                ]
                if not filtered_df.empty:
                    st.write("Antibiogram made with subgroup based on Speciality, Source, and Age Group.")
                else:
                    # Relax Speciality
                    filtered_df = df_antibiotics[
                        (df_antibiotics['Country'] == country) &
                        (df_antibiotics['Gender'] == gender) &
                        (df_antibiotics['Age.Group'] == age_group) &
                        (df_antibiotics['Source'] == source)
                    ]
                    if not filtered_df.empty:
                        st.write("Antibiogram made with subgroup based on Source and Age Group.")
                    else:
                        # Relax Source
                        filtered_df = df_antibiotics[
                            (df_antibiotics['Country'] == country) &
                            (df_antibiotics['Gender'] == gender) &
                            (df_antibiotics['Age.Group'] == age_group)
                        ]
                        if not filtered_df.empty:
                            st.write("Antibiogram made with subgroup based on Age Group.")
                        else:
                            # Relax Age Group
                            filtered_df = df_antibiotics[
                                (df_antibiotics['Country'] == country) &
                                (df_antibiotics['Gender'] == gender)
                            ]
                            if not filtered_df.empty:
                                st.write("Antibiogram made with subgroup based on Gender.")
                            else:
                                # Relax Gender
                                filtered_df = df_antibiotics[
                                    (df_antibiotics['Country'] == country)
                                ]
                                if not filtered_df.empty:
                                    st.write("Antibiogram made with subgroup based on Country.")
                                else:
                                    # Relax Country
                                    filtered_df = df_antibiotics.copy()
                                    if not filtered_df.empty:
                                        st.write("Antibiogram made with all available data.")
                                    else:
                                        st.write("No matching data found even after relaxing all criteria.")

            if not filtered_df.empty:
                # Ensure both 'Species', 'Antibiotic', and 'Resistance' columns are 1-dimensional and scalar
                if (filtered_df['Species'].apply(lambda x: isinstance(x, str)).all() and 
                    filtered_df['Antibiotic'].apply(lambda x: isinstance(x, str)).all() and 
                    filtered_df['Resistance'].apply(lambda x: isinstance(x, str)).all()):

                    # Calculate resistance counts and percentages by Species and Antibiotic
                    resistance_summary = filtered_df.groupby(['Species', 'Antibiotic', 'Resistance']).size().unstack(fill_value=0)
                    resistance_summary['Total Count'] = resistance_summary.sum(axis=1)
                    resistance_summary['% Susceptibility'] = (resistance_summary.get('Susceptible', 0) / resistance_summary['Total Count']) * 100
                    resistance_summary = resistance_summary.round({'% Susceptibility': 1})

                    # Format the percentage with a percentage sign
                    resistance_summary['% Susceptibility'] = resistance_summary['% Susceptibility'].apply(lambda x: f"{x:.1f}%")

                    # Filter for species with more than 30 total isolates
                    filtered_resistance_summary = resistance_summary[resistance_summary['Total Count'] > 30]

                    # Reshape the DataFrame so that each antibiotic has its own column
                    final_summary = filtered_resistance_summary.unstack(level='Antibiotic')['% Susceptibility']
                    final_summary.insert(0, 'Total Count', filtered_resistance_summary.groupby('Species')['Total Count'].first())

                    # Function to apply color formatting
                    def color_format(val):
                        if isinstance(val, str) and val.endswith('%'):
                            percentage = float(val.rstrip('%'))
                            if percentage > 75:
                                color = 'green'
                            elif 50 <= percentage <= 75:
                                color = 'orange'
                            else:
                                color = 'red'
                            return f'background-color: {color}'
                        return ''

                    # Apply the color formatting to the dataframe
                    styled_summary = final_summary.style.applymap(color_format, subset=pd.IndexSlice[:, final_summary.columns[1:]])

                    st.write("Detailed Antibiogram")
                    st.dataframe(styled_summary)
                else:
                    st.write("The 'Species', 'Antibiotic', or 'Resistance' column contains non-string data, which cannot be processed.")
            else:
                st.write("No data available for the selected criteria.")








        import streamlit as st
        import pandas as pd

        # Function to read the CSV file
        def load_data(file_path):
            # Load the dataset
            df = pd.read_csv(file_path, low_memory=False)
            # Replace spaces with dots in the column names
            df.columns = df.columns.str.replace(' ', '.')
            return df

        # Function to filter, convert, and calculate resistance percentage
        def filter_convert_and_calculate(df, antibiotics, antibiotic_prev):
            if antibiotic_prev not in df.columns:
                st.error(f"{antibiotic_prev} not found in the dataset columns.")
                return pd.DataFrame()
            
            # Filter rows where the selected previous antibiotic is marked as 'Resistant'
            resistant_df = df[df[antibiotic_prev] == 'Resistant']
            
            # Exclude the antibiotic of interest and convert to long format
            #long_df = resistant_df.drop(columns=[antibiotic_prev])
            long_df = pd.melt(resistant_df, 
                            id_vars=[col for col in resistant_df.columns if col not in antibiotics], 
                            value_vars=antibiotics, 
                            var_name='Antibiotics', 
                            value_name='Resistance')
            
            # Remove rows with NaN values in 'Resistance'
            long_df = long_df.dropna(subset=['Resistance'])
            
            return long_df

        # Function to generate the escalating antibiogram
        def generate_antibiogram(df_long, antibiotics):
            if df_long.empty:
                st.write("No data available for the selected criteria.")
                return
            
            resistance_summary = df_long.groupby(['Species', 'Antibiotics', 'Resistance']).size().unstack(fill_value=0)
            resistance_summary['Total Count'] = resistance_summary.sum(axis=1)
            resistance_summary['% Susceptibility'] = (resistance_summary.get('Susceptible', 0) / resistance_summary['Total Count']) * 100
            resistance_summary = resistance_summary.round({'% Susceptibility': 1})

            resistance_summary['% Susceptibility'] = resistance_summary['% Susceptibility'].apply(lambda x: f"{x:.1f}%")
            
            filtered_resistance_summary = resistance_summary[resistance_summary['Total Count'] > 30]

            final_summary = filtered_resistance_summary.unstack(level='Antibiotics')['% Susceptibility']
            final_summary.insert(0, 'Total Count', filtered_resistance_summary.groupby('Species')['Total Count'].first())

            def color_format(val):
                if isinstance(val, str) and val.endswith('%'):
                    percentage = float(val.rstrip('%'))
                    if percentage > 75:
                        color = 'green'
                    elif 50 <= percentage <= 75:
                        color = 'orange'
                    else:
                        color = 'red'
                    return f'background-color: {color}'
                return ''

            styled_summary = final_summary.style.applymap(color_format, subset=pd.IndexSlice[:, final_summary.columns[1:]])

            st.write("Escalating Antibiogram")
            st.dataframe(styled_summary)

        # Main Streamlit app code (continued from your existing app)

        # Place this code under the existing "Generate Escalating Antibiogram" button
        if st.button("Generate Escalating Antibiogram"):
            # Load the wide atlas data
            file_path = 'wide_atlas.csv'
            df = load_data(file_path)
            
            # Define the list of antibiotics for the analysis
            antibiotics = [
                'Amikacin', 'Amoxycillin.clavulanate', 'Ampicillin', 'Azithromycin', 'Cefepime', 
                'Cefoxitin', 'Ceftazidime', 'Ceftriaxone', 'Clarithromycin', 'Clindamycin', 
                'Erythromycin', 'Imipenem', 'Levofloxacin', 'Linezolid', 'Meropenem', 
                'Metronidazole', 'Minocycline', 'Penicillin', 'Piperacillin.tazobactam', 
                'Tigecycline', 'Vancomycin', 'Ampicillin.sulbactam', 'Aztreonam', 
                'Aztreonam.avibactam', 'Cefixime', 'Ceftaroline', 'Ceftaroline.avibactam', 
                'Ceftazidime.avibactam', 'Ciprofloxacin', 'Colistin', 'Daptomycin', 
                'Doripenem', 'Ertapenem', 'Gatifloxacin', 'Gentamicin', 'Moxifloxacin', 
                'Oxacillin', 'Quinupristin.dalfopristin', 'Sulbactam', 'Teicoplanin', 
                'Tetracycline', 'Trimethoprim.sulfa', 'Ceftolozane.tazobactam', 
                'Cefoperazone.sulbactam', 'Meropenem.vaborbactam', 'Cefpodoxime', 
                'Ceftibuten', 'Ceftibuten.avibactam', 'Tebipenem'
            ]
            
            # Use the antibiotic selected by the user in your app
            if 'antibiotic_prev' not in locals():
                antibiotic_prev = st.selectbox("Select Previous Antibiotic Used", antibiotics)
            
            # Filter, convert to long format, and generate antibiogram
            df_long = filter_convert_and_calculate(df, antibiotics, antibiotic_prev)
            generate_antibiogram(df_long, antibiotics)















    import re
    import pandas as pd
    import streamlit as st
    from Bio import Entrez, Medline

    # Define the function to extract the year from the source
    def extract_year(source):
        """
        Extract the year from the source string.
        
        :param source: Source string from which to extract the year.
        :return: Extracted year as a string.
        """
        match = re.search(r'\b\d{4}\b', source)
        return match.group(0) if match else None

    def remove_year_from_source(source):
        """
        Remove the year from the source string.
        
        :param source: Source string.
        :return: Source string without the year.
        """
        return re.sub(r'\b\d{4}\b', '', source).strip()
  
    


    # Function to search PubMed
    def search_pubmed(query, max_results=50):
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]

    # Function to fetch PubMed article details (Title, Authors, Abstract)
    def fetch_details(id_list):
        """
        Fetch detailed information for a list of PubMed IDs.
        
        :param id_list: List of PubMed IDs.
        :return: List of dictionaries containing Title, Authors, Abstract, Source, and Year.
        """
        ids = ",".join(id_list)
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
        records = list(Medline.parse(handle))
        handle.close()
        
        results = []
        for record in records:
            # Start by searching the title
            title = record.get("TI", "No title available")
            
            # Search for keywords (if available)
            keywords = record.get("OT", [])  # 'OT' is for Other Terms (Keywords)
            if keywords:
                keywords = ", ".join(keywords)
            else:
                keywords = "No keywords available"

            # Finally, search the body (abstract)
            abstract = record.get("AB", "No abstract available")
            
            # Extract other metadata
            authors = ", ".join(record.get("AU", ["No authors available"]))
            source = record.get("SO", "No source available")
            year = extract_year(source)
            
            results.append({
                "Title": title,
                "Keywords": keywords,
                "Abstract": abstract,
                "Authors": authors,
                "Source": source,
                "Year": year
            })
        
        return results
 
        
        
        
        
    with tab4:
        st.write("Search PubMed for related studies based on selected criteria.")

        # Create a search query from the selected criteria
        search_query = f"Infection AND {antibiotic} AND {source} AND {country}"
        st.write(f"Search Query: {search_query}")

        # Selection for the number of results to display
        max_results = st.selectbox("Select the number of results to display", options=[2, 5, 10, 20, 50], index=2)

        def progressive_search(query):
            pubmed_ids = search_pubmed(query, max_results=max_results)
            if pubmed_ids:
                return pubmed_ids, query

            search_criteria = [
                f"{antibiotic} OR {infection_diagnosis}",
                f"{antibiotic}",
            ]
            
            for crit in search_criteria:
                pubmed_ids = search_pubmed(crit, max_results=max_results)
                if pubmed_ids:
                    return pubmed_ids, crit
            
            return None, None

        if st.button("Search PubMed"):
            pubmed_ids, final_query = progressive_search(search_query)
            if pubmed_ids:
                pubmed_results = fetch_details(pubmed_ids)
                st.write(f"Results for query: {final_query}")
                for result in pubmed_results:
                    st.write(f"**Title:** {result['Title']}")
                    st.write(f"**Authors:** {result['Authors']}")
                    st.write(f"**Abstract:** {result['Abstract']}")
                    st.write("---")
            else:
                st.write("No studies found for the given search criteria.")


















    import re
    import pandas as pd
    import streamlit as st
    from Bio import Entrez, Medline

    # Define the function to extract the year from the source
    def extract_year(source):
        """
        Extract the year from the source string.
        
        :param source: Source string from which to extract the year.
        :return: Extracted year as a string.
        """
        match = re.search(r'\b\d{4}\b', source)
        return match.group(0) if match else None

    def remove_year_from_source(source):
        """
        Remove the year from the source string.
        
        :param source: Source string.
        :return: Source string without the year.
        """
        return re.sub(r'\b\d{4}\b', '', source).strip()

    # Your existing code

    with tab1:

        st.write("Generate a comprehensive report on the client's risk stratification and relevant research.")

        # Step 1: Filter data based on the most stringent criteria
        filtered_df = df_antibiotics[
            (df_antibiotics['Country'] == country) &
            (df_antibiotics['Gender'] == gender) &
            (df_antibiotics['Age.Group'] == age_group) &
            (df_antibiotics['Speciality'] == speciality) &
            (df_antibiotics['Source'] == source) &
            (df_antibiotics['Antibiotic'] == antibiotic) &
            (df_antibiotics['Infection_Diagnosis'] == infection_diagnosis)
        ]

        # Step 2: Relax criteria step by step if no exact match is found
        if filtered_df.empty:
            #st.write("Filtering search criteria using the demographic data......No exact match found. Showing best available match by relaxing criteria.")
            
            # Relax Infection Diagnosis
            filtered_df = df_antibiotics[
                (df_antibiotics['Country'] == country) &
                (df_antibiotics['Gender'] == gender) &
                (df_antibiotics['Age.Group'] == age_group) &
                (df_antibiotics['Speciality'] == speciality) &
                (df_antibiotics['Source'] == source) &
                (df_antibiotics['Antibiotic'] == antibiotic)
            ]
            if not filtered_df.empty:
                st.write(" ")
            else:
                # Relax Speciality
                filtered_df = df_antibiotics[
                    (df_antibiotics['Country'] == country) &
                    (df_antibiotics['Gender'] == gender) &
                    (df_antibiotics['Age.Group'] == age_group) &
                    (df_antibiotics['Source'] == source) &
                    (df_antibiotics['Antibiotic'] == antibiotic)
                ]
                if not filtered_df.empty:
                    st.write(" ")
                else:
                    # Relax Source
                    filtered_df = df_antibiotics[
                        (df_antibiotics['Country'] == country) &
                        (df_antibiotics['Gender'] == gender) &
                        (df_antibiotics['Age.Group'] == age_group) &
                        (df_antibiotics['Antibiotic'] == antibiotic)
                    ]
                    if not filtered_df.empty:
                        st.write(" ")
                    else:
                        # Relax Age Group
                        filtered_df = df_antibiotics[
                            (df_antibiotics['Country'] == country) &
                            (df_antibiotics['Gender'] == gender) &
                            (df_antibiotics['Antibiotic'] == antibiotic)
                        ]
                        if not filtered_df.empty:
                            st.write(" ")
                        else:
                            # Relax Gender
                            filtered_df = df_antibiotics[
                                (df_antibiotics['Country'] == country) &
                                (df_antibiotics['Antibiotic'] == antibiotic)
                            ]
                            if not filtered_df.empty:
                                st.write(" ")
                            else:
                                # Relax Country
                                filtered_df = df_antibiotics[
                                    (df_antibiotics['Antibiotic'] == antibiotic)
                                ]
                                if not filtered_df.empty:
                                    st.write(" ")
                                else:
                                    st.write(" ")

        # Step 3: Generate the summary based on the available data
        if not filtered_df.empty:
            # Count the occurrences of each Species
            species_counts = filtered_df['Species'].value_counts().head(10)

            resistance_levels = []

            for i, species in enumerate(species_counts.index):
                resistance_data = filtered_df[filtered_df['Species'] == species]['Resistance'].value_counts(normalize=True) * 100
                
                # Calculate the resistance percentage
                resistant_percentage = resistance_data.get('Resistant', 0)
                resistance_levels.append((species, resistant_percentage))

            # Generate the summary
            summary = f"This is a summary of the surveillance data for {antibiotic} based on the variables selected: age group {age_group}, gender {gender}, country {country}, speciality {speciality}, and source {source}. "
            summary += "The surveillance results show that the top 10 most common bacteria are "
            summary += ", ".join([f"{species} with resistance levels of {percentage:.2f}%" for species, percentage in resistance_levels]) + "."

            # Check for Staphylococcus aureus and calculate MRSA percentage if applicable
            if 'Staphylococcus aureus' in [species for species, _ in resistance_levels]:
                staph_aureus_df = filtered_df[(filtered_df['Species'] == 'Staphylococcus aureus') & (filtered_df['Antibiotic'] == 'Oxacillin')]
                if not staph_aureus_df.empty:
                    total_staph_aureus = len(staph_aureus_df)
                    mrsa_count = staph_aureus_df[staph_aureus_df['Resistance'] == 'Resistant'].shape[0]
                    mrsa_percentage = (mrsa_count / total_staph_aureus) * 100
                    summary += f"\n\nAmong the *Staphylococcus aureus* isolates, {mrsa_percentage:.1f}% are methicillin-resistant *S. aureus* (MRSA) based on their resistance to Oxacillin."

        else:
            summary = "No surveillance data summary could be generated due to insufficient data."

        # Replicate tab3: Search PubMed for related studies
        search_query = f"Infection, {antibiotic} AND {infection_diagnosis} AND  {country}"

        pubmed_ids, final_query = progressive_search(search_query)

        # Limit the number of abstracts to 50
        if pubmed_ids:
            pubmed_ids = pubmed_ids[:50]  # Only take the first 50 IDs
            pubmed_results = fetch_details(pubmed_ids)
            abstracts_tab3 = "\n\n".join([result['Abstract'] for result in pubmed_results])
        else:
            abstracts_tab3 = "No studies found for the given search criteria."

        # Combine the summary from tab1 and the abstracts from tab3
        combined_input = summary + "\n\n" + abstracts_tab3

        # Step 4: Prepare the data for the CSV
        abstracts_data = []
        for result in pubmed_results:
            title = result.get('Title', '')  # Get the title of the article
            abstract_text = result.get('Abstract', '')
            authors = "".join(result.get('Authors', []))  # Join authors as a string
            source = result.get('Source', '')
            year = extract_year(source)
            source_without_year = remove_year_from_source(source)

            abstracts_data.append({
                'title': title,          # Add the title to the data dictionary
                'text': abstract_text,
                'authors': authors,
                'source': source,
                'year': year
            })

        # Step 5: Create a DataFrame
        abstracts_df = pd.DataFrame(abstracts_data)



        # Step 6: Output the DataFrame as a CSV file
        abstracts_df.to_csv('abstracts_output.csv', index=False)
        #st.write("Abstracts have been saved as 'abstracts_output.csv'.")















        import os
        import re
        import pandas as pd
        import streamlit as st
        from dotenv import load_dotenv
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.vectorstores import FAISS
        from langchain.storage import LocalFileStore
        from langchain.docstore.document import Document
        from openai import OpenAI  # Import the synchronous OpenAI client

        # Load environment variables from the .env file
        load_dotenv()

        # Initialize the local file store for caching
        store = LocalFileStore("./cache/")

        # Initialize the OpenAI embedding model
        openai_api_key = os.getenv("OPENAI_API_KEY")
        embedder = OpenAIEmbeddings(api_key=openai_api_key)

        # Step 1: Load the CSV file
        csv_path = 'abstracts_output.csv'  # Replace with your actual CSV file path
        df = pd.read_csv(csv_path)

        # Step 2: Prepare documents and metadata
        texts = df['text'].tolist()
        metadata = [{"authors": row['authors'], "title": row['title'], "source": row['source'], "year": row['year']} for _, row in df.iterrows()]

        # Convert each text and its metadata into a Document object
        documents = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadata)]

        # Step 3: Create the FAISS vector store from the documents and cache-backed embedder
        vector_store = FAISS.from_documents(documents, embedder)

        # Initialize the OpenAI client
        client = OpenAI(api_key=openai_api_key)

        # Step 4: Use the vector store to retrieve the top 5 relevant documents
        query = "Summarize the most relevant abstracts"  # Replace this with your actual query
        top_k_documents = vector_store.similarity_search(query, k=5)

        # Step 5: Use OpenAI to generate the report using RAG within Streamlit
        if st.button("Generate Report"):
            try:
                # Summarize the surveillance data in bullet points
                bullet_summary_prompt = f"Summarize the following in bullet points:\n{summary}"
                bullet_summary_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": bullet_summary_prompt}],
                    max_tokens=300,
                    temperature=0.1
                )
                bullet_summary = bullet_summary_response.choices[0].message.content.strip()

                final_report = "### AI Summary of Surveillance Data\n" + bullet_summary + "\n\n"

                references = []

                for doc in top_k_documents:
                    text = doc.page_content
                    meta = doc.metadata

                    # Prepare the prompt for summarization
                    summary_prompt = "You are helping doctors use research to determine if the antibiotic is approriate for the infection, summarize the following text in a concise paragraph form, ensuring that all sentences are complete and properly structured:"
                    summary_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": f"{summary_prompt}\n{text}"}],
                        max_tokens=100,
                        temperature=0.1
                    )
                    summary_result = summary_response.choices[0].message.content.strip()

                    # Ensure the summary ends with a complete sentence by avoiding any dangling punctuation
                    summary_result = re.sub(r'\.\s*$', '', summary_result)

                    # Extract metadata for APA citation
                    authors = meta['authors'].split(", ")
                    year = meta['year']
                    title = meta.get('title', 'Untitled')
                    source = meta.get('source', '')

                    if len(authors) == 1:
                        citation_in_text = f"({authors[0]}, {year})"
                    elif len(authors) == 2:
                        citation_in_text = f"({authors[0]} & {authors[1]}, {year})"
                    else:  # Three or more authors
                        citation_in_text = f"({authors[0]} et al., {year})"

                    # Format the full reference in APA style with the title included
                    if len(authors) == 1:
                        citation_full = f"{authors[0]}. ({year}). {title}. {source}."
                    elif len(authors) == 2:
                        citation_full = f"{authors[0]} & {authors[1]}. ({year}). {title}. {source}."
                    else:
                        citation_full = f"{authors[0]}, {authors[1]}, & {authors[2]}. ({year}). {title}. {source}."

                    # Add the full reference to the references list
                    references.append(citation_full)

                    # Combine the summary and the APA citation, and put a full stop after the citation
                    paragraph = f"{summary_result} {citation_in_text}."

                    # Append this paragraph to the final report
                    final_report += paragraph + "\n\n"

                # Add the references section to the final report
                final_report += "\n\n### References\n"
                for ref in references:
                    final_report += f"{ref}\n\n"

                # Display the generated report
                st.write("### Generated Report")
                st.write(final_report)

            except Exception as e:
                st.error(f"Error generating report: {e}")
















