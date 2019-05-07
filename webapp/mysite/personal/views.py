from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    return render(request, "personal/home.html")

def contact(request):
    return render(request, "personal/basic.html", {"contact_page_content" : "This is the content of the contact page."})
