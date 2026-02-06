from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required  
from .ai_logic import CrackDetector
import os

# Initialize the detector once
detector = CrackDetector()

@login_required  
def index(request):
    return render(request, 'detector/index.html')

@login_required  
def process_image_api(request):
    # API is also protected
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        
        filename = fs.save(uploaded_file.name, uploaded_file)
        input_path = fs.path(filename)
        
        output_filename = f"result_{filename}"
        output_path = os.path.join(fs.location, output_filename)
        
        results, error = detector.process_image(input_path, output_path)
        
        if error:
            return JsonResponse({'success': False, 'error': error})
        
        return JsonResponse({
            'success': True,
            'original_url': fs.url(filename),
            'result_url': fs.url(output_filename),
            'data': results
        })
        
    return JsonResponse({'success': False, 'error': 'No image provided'})