# SER_API

SER_API is a Python API that can detect emotion in from an audio file.

The emotion can be detected are:
- Happy
- Angry
- Sad

The audio file must be on .wav format.

Follow these steps to install and run the project:

## Installation
### Prerequisites

Ensure you have the following installed on your system:

- Python 3.6 or later
- pip, the Python package installer

### Steps

1. **Clone the repository**

    First, clone the repository to your local machine using git. Open your terminal and run:

    ```bash
    git clone https://github.com/MarieLePanda/SER_API.git
    ```

2. **Navigate to the project directory**

    Change your current directory to the project's directory with:

    ```bash
    cd SER_API
    ```

3. **Create a virtual environment (Optional but recommended)**

    It's recommended to create a virtual environment to keep the dependencies required by this project separate from your global Python environment. Here's how you can create one:

    ```bash
    python3 -m venv env
    ```

4. **Activate the virtual environment**

    Before installing the dependencies, activate the virtual environment:

    - On Windows:
        ```bash
        .\env\Scripts\activate
        ```
    - On Unix or MacOS:
        ```bash
        source env/bin/activate
        ```

5. **Install the dependencies**

    Use pip to install the project dependencies from the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

6. **Run the application**

    Now, you're ready to run the application:

    ```bash
    python app.py
    ```

That's it! If everything is set up correctly, you should now be able to access the API at `localhost:5000` (or whatever host and port you've set).

Remember to replace `python3` and `pip` with `python` or `pip3` if you're using a different version.

## Usage
To send an audio file to the API, you can use a POST request with the audio file attached in the form data. 
```Bash
curl -X POST -F "file=@path_to_your_file.wav" http://localhost:5000/get_sentiment
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.


## License

[MIT](https://choosealicense.com/licenses/mit/)
