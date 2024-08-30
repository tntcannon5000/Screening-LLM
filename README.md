# Fully Automated Screening Interview & Performance Evaluation System

## Key Features

This Fully Automated Screening Interview & Performance Evaluation System offers several features:

* **Speech-to-Speech Interaction:**  Engage candidates in a natural and conversational interview experience through seamless speech-to-speech interaction powered by cutting-edge Large Language Models (LLMs).

* **Automated Relevant Questioning:**  The system intelligently generates and asks relevant interview questions tailored to the specific job requirements, ensuring a comprehensive assessment of the candidate's skills and qualifications.

* **Multi-LLM Evaluation Process:**  A sophisticated ensemble of LLMs works in tandem to analyze candidate responses in real-time, providing a comprehensive and unbiased evaluation based on sentiment analysis, accuracy assessment, and other relevant factors.

* **Difficulty-Based Pass/Fail Recommendation:**  The system goes beyond a simple pass/fail decision by offering a nuanced recommendation based on the difficulty level of the interview and the candidate's performance. This allows for a more accurate and insightful assessment of candidate suitability.
![system_design](https://github.com/user-attachments/assets/8f4928c2-3dc6-422b-9a06-a9a8cf9835e4)



## Installation

**Prerequisites:**

* **Miniconda or Anaconda:** You'll need to have either Miniconda (a minimal installer for conda) or Anaconda (a full distribution including conda and many popular packages) installed on your system.
  * **Download Miniconda:** [[Miniconda download page]](https://docs.anaconda.com/miniconda/)
  * **Download Anaconda:** [[Anaconda download page]](https://docs.anaconda.com/anaconda/install/)

* **Conda:** Make sure you can access the conda terminal by either finding it in your start menu (Windows) or calling it from bash/cmd
  ![image](https://github.com/user-attachments/assets/314eaa20-3fa1-4a54-bc70-82721beaaeba)


**Steps:**

1. **Clone or Download the Project:**
   - If you have `git` installed, you can clone the repository:
     ```bash
     git clone https://github.com/tntcannon5000/Screening-LLM/
     ```
   - Otherwise, download the project as a zip file using the download button, or from releases and extract it to a suitable location.

2. **Navigate to the Project Directory:**
   - Open your terminal or command prompt and use the `cd` command to navigate to the directory where you extracted the project files.


3. **Create the Conda Environment:**
   - Run the following command to create the conda environment using the provided `environment.yml` file:
     ```bash
     conda env create -f environment.yml
     ```
   - This will install all the necessary packages and dependencies for the project
     ![image](https://github.com/user-attachments/assets/97d2f836-9bf6-40e3-9445-65bb007b1ce0)


4. **Activate the Environment:**
   - Activate the newly created environment using the following command:
     ```bash
     conda activate interviewsystem
     ```
   - You should see the following to indicate you've successfully activated the virtual environment
     ![image](https://github.com/user-attachments/assets/76da2ff8-7fa6-4a61-b444-c7df69f95658)

5. **Provide a CV for the system**
   - Place a CV

   ![image](https://github.com/user-attachments/assets/1be020da-0c41-434c-a56a-8bd289c7db15)

6. **Run the Main Script (or other scripts):**
   - You can now run your main script or other scripts in your project. For example:
     ```bash
     python main.py  # Replace 'main.py' with the actual name of your script
     ```

   ![image](https://github.com/user-attachments/assets/1be020da-0c41-434c-a56a-8bd289c7db15)



## License

This project is licensed under the **GNU General Public License (GPL)** - see the [LICENSE](LICENSE) file for details.


## Contact

To contact us, feel free to reach out to us via LinkedIn, available on our profiles.
