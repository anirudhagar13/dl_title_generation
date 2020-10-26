from django.template import loader
from django.shortcuts import render
from django.http import HttpResponse

from .backend import generate

# Create your views here.
def index(request):
	context = dict()
	template = loader.get_template('main/index.html')
	if request.method == 'POST':
		abstract = request.POST.get('abstract')
		print ('Abstract received')
		
		# title generation
		title = generate(abstract)
		if not title:
			title = '<title not generated>'

		context = {'abstract':abstract, 'title':title}

	return HttpResponse(template.render(context, request))
	