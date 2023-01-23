# choreography-analysis
choreography analysis using Mediapipe and Librosa

## How to Use
1. Create and source local environment: `python3 -m venv venv && source venv/bin/activate`
2. Install necessary packages: `pip3 install -r requirements.txt`
3. Add any dance video in the `media/videos/` and insert the relative video url in `test.py` 
4. Run the script on your terminal by `python test.py` . This should generate frames from the video and annotate and give you basic audio insights.
5. Replace images url path on `test_image_diff.py` to calculate the image differences of one movement to another
