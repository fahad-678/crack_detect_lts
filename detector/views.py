from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .ai_logic import CrackDetector
import os

# Initialize the detector once to save memory
detector = CrackDetector()

def index(request):
    context = {}
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        
        # Save the uploaded file
        filename = fs.save(uploaded_file.name, uploaded_file)
        input_path = fs.path(filename)
        
        # Define the output path for the processed image
        output_filename = f"result_{filename}"
        output_path = os.path.join(fs.location, output_filename)
        
        # Run Detection
        results, error = detector.process_image(input_path, output_path)
        
        if error:
            context['error'] = error
        else:
            context['results'] = results
            context['original_url'] = fs.url(filename)
            context['result_url'] = fs.url(output_filename)
            
    return render(request, 'detector/index.html', context)