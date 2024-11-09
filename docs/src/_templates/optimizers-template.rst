{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block image %}

    {% set image_path = fullname.split('.') %}
    {% set image_path = image_path[:-1] + ['../../_static/media', 'rosenbrock_' + image_path[-1] + '.png'] %}
    {% set image_path = '/'.join(image_path) %}
    {% if image_path %}
    .. image:: {{ image_path }}
        :align: center
    {% endif %}

   {% endblock %}

   {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      {%- if item not in inherited_members %}
         ~{{ name }}.{{ item }}
      {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
