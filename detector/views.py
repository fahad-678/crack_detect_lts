from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required  
from .ai_logic import CrackDetector
import os

# Initialize the detector once
detector = CrackDetector()

def home(request):
    return render(request, 'detector/home.html')

@login_required  
def demo(request):
    return render(request, 'detector/demo.html')

def instructions(request):
    return render(request, 'detector/instructions.html')

def samples(request):
    return render(request, 'detector/samples.html')

def about(request):
    return render(request, 'detector/about.html')

@login_required  
def process_image_api(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        
        try:
            conf = float(request.POST.get('conf', 0.25))
            iou = float(request.POST.get('iou', 0.5))
            coin_diameter = float(request.POST.get('coin_diameter', 18.5))
            mode = request.POST.get('mode', 'crack_only') # Extracting mode
        except ValueError:
            return JsonResponse({'success': False, 'error': 'Invalid parameter values'})

        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        input_path = fs.path(filename)
        
        output_filename = f"result_{filename}"
        output_path = os.path.join(fs.location, output_filename)
        
        # Pass the mode to the detector
        results, error = detector.process_image(
            input_path, output_path, 
            conf=conf, iou=iou, coin_diameter=coin_diameter, mode=mode
        )
        
        if error:
            return JsonResponse({'success': False, 'error': error})
        
        return JsonResponse({
            'success': True,
            'filename': uploaded_file.name,
            'original_url': fs.url(filename),
            'result_url': fs.url(output_filename),
            'data': results
        })
        
    return JsonResponse({'success': False, 'error': 'No image provided'})