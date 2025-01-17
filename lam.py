<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Git Operations Example</title>
    <style>
        /* Body Styling */
        body {
            background-color: #ffffff; /* White background */
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            font-family: Arial, sans-serif;
        }

        /* Container Styling */
        .container {
            text-align: center;
            width: 90%;
            max-width: 600px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
            background-color: #f9f9f9; /* Slightly gray background */
        }

        /* Button Styling */
        .button {
            background-color: #4a4a4a; /* Dark gray button, closer to ChatGPT style */
            border: none;
            color: white;
            padding: 15px 30px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin-bottom: 20px;
            cursor: pointer;
            border-radius: 5px;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }

        .button:hover {
            background-color: #333333;
            transform: translateY(-2px);
        }

        .button:active {
            transform: translateY(0);
        }

        .button:disabled {
            background-color: #a0a0a0;
            cursor: not-allowed;
        }

        /* Textarea Styling */
        .textarea {
            width: 100%;
            height: 300px;
            padding: 10px;
            box-sizing: border-box;
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: #e0e0e0; /* Gray background */
            resize: none;
            font-size: 16px;
            color: #333;
            overflow: auto;
            font-family: Consolas, "Courier New", monospace;
        }

        /* Responsive Design */
        @media (max-width: 600px) {
            .button {
                width: 100%;
                padding: 15px;
                font-size: 18px;
            }

            .textarea {
                height: 200px;
                font-size: 14px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <button class="button" id="giveCodeButton">Give Me Code</button>
        <textarea class="textarea" id="content" readonly>
# Clone the repository
git clone https://github.com/your-repo.git

# Navigate into the repository directory
cd your-repo

# Create a new branch
git checkout -b feature-branch

# Make your code changes

# Add changes to the staging area
git add .

# Commit your changes
git commit -m "Add new feature"

# Push the branch to the remote repository
git push origin feature-branch

# Create a Pull Request on GitHub
        </textarea>
    </div>

    <script>
        // Event listener for the "Give Me Code" button
        document.getElementById('giveCodeButton').addEventListener('click', async function() {
            const button = this;
            const backendUrl = '/xxx'; // Replace with your backend URL if different

            // Disable the button to prevent multiple clicks
            button.disabled = true;
            const originalText = button.textContent;
            button.textContent = 'Sending...';

            try {
                // Send a POST request to the backend
                const response = await fetch(backendUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        // Add any necessary data to send with the request
                        action: 'get_code'
                    })
                });

                if (!response.ok) {
                    throw new Error(`Server responded with status ${response.status}`);
                }

                const data = await response.json();

                // Check if the JSON contains the downloadUrl
                if (data.downloadUrl) {
                    // Create a temporary anchor element to initiate the download
                    const downloadLink = document.createElement('a');
                    downloadLink.href = data.downloadUrl;
                    downloadLink.download = ''; // Let the browser decide the filename
                    document.body.appendChild(downloadLink);
                    downloadLink.click();
                    document.body.removeChild(downloadLink);
                } else {
                    throw new Error('downloadUrl not found in the response.');
                }
            } catch (error) {
                console.error('Request failed:', error);
                alert(`Request failed: ${error.message}`);
            } finally {
                // Re-enable the button and restore its original text
                button.disabled = false;
                button.textContent = originalText;
            }
        });
    </script>
</body>
</html>
